// Copyright (C) 2011-2012 the original author or authors.
// See the LICENCE.txt file distributed with this work for additional
// information regarding copyright ownership.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package demo

import org.tensorflow.{DataType, Graph, Output, Session, Tensor}
import spray.json._
import DefaultJsonProtocol._

import java.io.IOException
import java.nio.charset.Charset
import java.nio.file.{Files, Path, Paths}
import java.util.{Arrays, List}

object LabelImageUtils {
  private def constructAndExecuteGraphToNormalizeInceptionImage = true
  def constructAndExecuteGraphToNormalizeInceptionImage(imageBytes: Array[Byte]): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)

    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val H: Int = 224
    val W: Int = 224
    val mean: Float = 117f
    val scale: Float = 1f

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    val input: Output = b.constant("input", imageBytes)
    val output: Output =
      b.div(
        b.sub(
          b.resizeBilinear(
          b.expandDims(
            b.cast(b.decodeJpeg(input, 3), DataType.FLOAT),
            b.constant("make_batch", 0)),
          b.constant("size", Array[Int](H, W))),
          b.constant("mean", mean)),
        b.constant("scale", scale))
    var s: Session = new Session(g)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    s.close
    g.close
    result
  }

  private def executeInceptionGraph = true
  def executeInceptionGraph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    executePreTrainedGraph(graphDef, image, "input", "output")
  }

  private def executeRasterVisionTaggingGraph = true
  def executeRasterVisionTaggingGraph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    executePreTrainedGraph(graphDef, image, "input_1", "dense/Sigmoid")
  }

  private def executePreTrainedGraph = true
  def executePreTrainedGraph(graphDef: Array[Byte], image: Tensor, inputOp: String, outputOp: String): Array[Float] = {
    var g: Graph = new Graph
    g.importGraphDef(graphDef)
    var s: Session = new Session(g)
    var result: Tensor = s.runner().feed(inputOp, image).fetch(outputOp).run().get(0)
    s.close
    g.close
    val rshape: Array[Long] = result.shape
    val rshapeString: String = Arrays.toString(rshape)
    if (result.numDimensions != 2 || rshape(0) != 1) {
      throw new RuntimeException(
        String.format(
          f"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape $rshapeString%s"))
    }
    val nlabels: Int = rshape(1).asInstanceOf[Int]
    result.copyTo(Array.ofDim[Float](1, nlabels))(0)
  }

  private def maxIndex = true
  def maxIndex(probabilities: Array[Float]): Int = {
    var best: Int = 0
    val i: Int = 1
    for (i <- 1 to probabilities.length-1) {
      if (probabilities(i) > probabilities(best)) {
        best = i
      }
    }
    best
  }

  private def readAllBytesOrExit = true
  def readAllBytesOrExit(path: Path): Array[Byte] = {
    try {
      return Files.readAllBytes(path)
    } catch {
      case e: IOException => {
        System.err.println("Failed to read [" + path + "]: " + e.getMessage)
        System.exit(1)
      }
    }
    null
  }

  private def readAllLinesOrExit = true
  def readAllLinesOrExit(path: Path): List[String] = {
    try {
      return Files.readAllLines(path, Charset.forName("UTF-8"))
    } catch {
      case e: IOException => {
        System.err.println("Failed to read [" + path + "]: " + e.getMessage)
        System.exit(0)
      }
    }
    null
  }

  private def constructAndExecuteGraphToNormalizeRasterVisionImage = true
  def constructAndExecuteGraphToNormalizeRasterVisionImage(imagePathString: String): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)
    // Task: normalize images using channel_stats.json file for the dataset
    // Maybe repetitive/too many calls
    val rasterVisionDataDir = sys.env("RASTER_VISION_DATA_DIR")
    val datasetDir = Paths.get(rasterVisionDataDir, "datasets").toString()
    val planetKaggleDatasetPath = Paths.get(datasetDir, "planet_kaggle").toString()
    val planetKaggleDatasetStatsPath = Paths.get(planetKaggleDatasetPath, "planet_kaggle_jpg_channel_stats.json").toString()
    // Maybe repetitive open/read/close json pattern
    val source: scala.io.Source = scala.io.Source.fromFile(planetKaggleDatasetStatsPath)
    val lines: String = try source.mkString finally source.close
    val stats: Map[String, Array[Float]] = lines.parseJson.convertTo[Map[String, Array[Float]]]
    val means: Array[Float] = stats("means")
    val stds: Array[Float] = stats("stds")

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    var imageTensor: Tensor = b.decodeWithMultibandTile(imagePathString)
    val input: Output = b.constantTensor("input", imageTensor)
    imageTensor.close

    val shape: Array[Long] = imageTensor.shape
    val height: Int = shape(0).asInstanceOf[Int]
    val width: Int = shape(1).asInstanceOf[Int]
    val channels: Int = shape(2).asInstanceOf[Int]
    val meansArray: Array[Array[Array[Float]]] = Array.ofDim(height, width, channels)
    val stdsArray: Array[Array[Array[Float]]] = Array.ofDim(height, width, channels)

    // build 3D matrices where each 2D layer is ones(height, width) * the respective channel statistic
    for (h <- 0 to height - 1) {
       for (w <- 0 to width - 1) {
         for (c <- 0 to channels - 1) {
           meansArray(h)(w)(c) = means(c)
           stdsArray(h)(w)(c) = stds(c)
         }
       }
    }
    var meansTensor: Tensor = Tensor.create(meansArray)
    var stdsTensor: Tensor = Tensor.create(stdsArray)
    val meansOutput: Output = b.constantTensor("means", meansTensor)
    val stdsOutput: Output = b.constantTensor("stds", stdsTensor)
    meansTensor.close
    stdsTensor.close

    val output: Output =
      b.div(
        b.sub(
          b.expandDims(
            b.cast(input, DataType.FLOAT),
            b.constant("make_batch", 0)),
          meansOutput),
        stdsOutput)

    var s: Session = new Session(g)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    s.close
    g.close
    result
  }

  def getExperimentDir(runName: String): String = {
    // The RASTER_VISION_DATA_DIR environment variable must be set to locate files.
    val rasterVisionDataDir = sys.env("RASTER_VISION_DATA_DIR")
    val resultsDir = Paths.get(rasterVisionDataDir, "results").toString()
    val experimentDir = Paths.get(resultsDir, runName).toString()
    experimentDir
  }

  def getGraphPath(runName: String): Path = {
    val experimentDir = getExperimentDir(runName)

    // Convention from code that writes frozen graph to experiment directory.
    val graphName = runName.replace('/', '_') + "_graph.pb"
    Paths.get(experimentDir, graphName)
  }

  def printBestMatch(labelProbabilities: Array[Float], labels: List[String]) {
    val bestLabelIdx: Int = LabelImageUtils.maxIndex(labelProbabilities)
    val bestLabel: String = labels.get(bestLabelIdx)
    val bestLabelLikelihood: Float = labelProbabilities(bestLabelIdx) * 100f
    println(f"BEST MATCH: $bestLabel%s ($bestLabelLikelihood%.2f%% likely)")
  }

  def printMatches(runName: String, labelProbabilities: Array[Float], labels: List[String]) {
    // Task: Use thresholds to do multi-label classification
    val experimentDir: String = getExperimentDir(runName)
    val thresholdsPath: String = Paths.get(experimentDir, "thresholds.json").toString()
    val source: scala.io.Source = scala.io.Source.fromFile(thresholdsPath)
    val lines: String = try source.mkString finally source.close
    val thresholds: Array[Float] = lines.parseJson.convertTo[Array[Float]]
    var i: Int = 0
    for (i <- 0 to labels.size - 1) {
      val labelProbability: Float = labelProbabilities(i) * 100f
      val threshold: Float = thresholds(i) * 100f
      val label: String = labels.get(i)
      if (labelProbability >= threshold) {
        print(f"$label%s ")
      }
    }
    println()
    i = 0
    for (i <- 0 to labels.size - 1) {
      val labelProbability: Float = labelProbabilities(i) * 100f
      val threshold: Float = thresholds(i) * 100f
      val label: String = labels.get(i)
      if (labelProbability >= threshold) {
        println(f"MATCH: $label%s ($labelProbability%.2f%% likely)")
      }
    }
  }
}
