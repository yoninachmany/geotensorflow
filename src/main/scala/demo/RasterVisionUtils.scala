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

object RasterVisionUtils {
  private def executeRasterVisionTaggingGraph = true
  def executeRasterVisionTaggingGraph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    LabelImageUtils.executePreTrainedGraph(graphDef, image, "input_1", "dense/Sigmoid")
  }

  def readChannelStats: Map[String, Array[Double]] = {
    val rasterVisionDataDir = sys.env("RASTER_VISION_DATA_DIR")
    val datasetDir = Paths.get(rasterVisionDataDir, "datasets").toString()
    val planetKaggleDatasetPath = Paths.get(datasetDir, "planet_kaggle").toString()
    val planetKaggleDatasetStatsPath = Paths.get(planetKaggleDatasetPath, "planet_kaggle_jpg_channel_stats.json").toString()
    val source: scala.io.Source = scala.io.Source.fromFile(planetKaggleDatasetStatsPath)
    val lines: String = try source.mkString finally source.close
    val stats: Map[String, Array[Double]] = lines.parseJson.convertTo[Map[String, Array[Double]]]
    stats
  }

  private def constructAndExecuteGraphToNormalizeRasterVisionImage = true
  def constructAndExecuteGraphToNormalizeRasterVisionImage(imagePathString: String): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    var imageTensor: Tensor = b.decodeJpegGeoTrellis(imagePathString)
    val input: Output = b.constantTensor("input", imageTensor)

    // Task: normalize images using channel_stats.json file for the dataset
    val stats: Map[String, Array[Double]] = readChannelStats
    val means: Array[Double] = stats("means")
    val stds: Array[Double] = stats("stds")
    val shape: Array[Long] = imageTensor.shape
    val height: Int = shape(0).asInstanceOf[Int]
    val width: Int = shape(1).asInstanceOf[Int]
    val channels: Int = shape(2).asInstanceOf[Int]
    val meansArray: Array[Array[Array[Double]]] = Array.ofDim(height, width, channels)
    val stdsArray: Array[Array[Array[Double]]] = Array.ofDim(height, width, channels)

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

    val output: Output =
      b.castFloat(
        b.div(
          b.sub(
            b.expandDims(
              b.cast(input, DataType.DOUBLE),
              b.constant("make_batch", 0)),
            b.constantTensor("means", meansTensor)),
          b.constantTensor("stds", stdsTensor)))
    val s: Session = new Session(g)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    result
    // try {
    //   result
    // }
    // finally {
    //   g.close
    //   imageTensor.close
    //   meansTensor.close
    //   stdsTensor.close
    //   s.close
    // }
  }

  private def constructAndExecuteGraphToNormalizeInGeoTrellisRasterVisionImage = true
  // def constructAndExecuteGraphToNormalizeRasterVisionImage(imagePathString: String): Tensor = {
  def constructAndExecuteGraphToNormalizeInGeoTrellisRasterVisionImage(imagePathString: String): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)
    // Task: normalize images using channel_stats.json file for the dataset
    // Maybe repetitive/too many calls
    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    var imageTensor: Tensor = b.decodeAndNormalizeJpegGeoTrellis(imagePathString)
    val input: Output = b.constantTensor("input", imageTensor)
    val shape: Array[Long] = imageTensor.shape

    val output: Output =
        b.expandDims(
          b.cast(input, DataType.FLOAT),
          b.constant("make_batch", 0))
    val s: Session = new Session(g)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    result
    // try {
    //   result
    // }
    // finally {
    //   g.close
    //   imageTensor.close
    //   meansTensor.close
    //   stdsTensor.close
    //   s.close
    // }
  }

  def getExperimentDir(runName: String): String = {
    // The RASTER_VISION_DATA_DIR environment variable must be set to locate files.
    val rasterVisionDataDir = sys.env("RASTER_VISION_DATA_DIR")
    val resultsDir = Paths.get(rasterVisionDataDir, "results").toString()
    val experimentDir = Paths.get(resultsDir, runName).toString()
    experimentDir
  }

  def getGraphPath(runName: String): Path = {
    Paths.get(getExperimentDir(runName), "output_graph.pb")
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
      // if (labelProbability >= threshold) {
        println(f"MATCH: $label%15s $labelProbability%15.2f%% likely $threshold%15.2f%% likely")
      // }
    }
  }
}
