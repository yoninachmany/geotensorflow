/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License")
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package demo

import geotrellis.raster.{Tile, MultibandTile}
import geotrellis.raster.io.geotiff.MultibandGeoTiff
import geotrellis.raster.io.geotiff.reader.GeoTiffReader
import org.tensorflow.{DataType, Graph, Output, Session, Tensor, TensorFlow, Shape}
import spray.json._
import DefaultJsonProtocol._

import java.io.{IOException, PrintStream}
import java.nio.ByteBuffer
import java.nio.charset.Charset
import java.nio.file.{Files, Path, Paths}
import java.util.{Arrays, List}

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
object LabelImageRasterVision {
  def printUsage(s: PrintStream) {
    val url: String =
      "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    s.println(
      "Scala program that uses a model trained by Raster Vision (https://github.com/azavea/raster-vision)")
    s.println("to label Tiff images.")
    s.println("TensorFlow version: " + TensorFlow.version)
    s.println
    s.println("Usage: label_image <run namer> <image file>")
    s.println
    s.println("Where:")
    s.println("<run name> is the unique ID for the experiment")
    s.println("<image file> is the path to a JPEG image file")
  }

  def main(args: Array[String]) {
    if (args.length != 2) {
      printUsage(System.err)
      System.exit(1)
    }
    val runName: String = args(0)
    val imageFile: String = args(1)

    // The RASTER_VISION_DATA_DIR environment variable must be set, to locate files.
    val rasterVisionDataDir = sys.env("RASTER_VISION_DATA_DIR")
    val resultsDir = Paths.get(rasterVisionDataDir, "results").toString()
    val experimentDir = Paths.get(resultsDir, runName).toString()

    // Convention from code that writes frozen graph to experiment directory.
    val graphName = runName.replace('/', '_') + "_graph.pb"
    val graphDef: Array[Byte] = readAllBytesOrExit(Paths.get(experimentDir, graphName))
    // Easiest way to access labels
    val labels: List[String] = readAllLinesOrExit(Paths.get("labels.txt"))
    val imagePathString: String = Paths.get(imageFile).toString

    // If normalization happens in GeoTrellis, remove
    var image: Tensor = constructAndExecuteGraphToNormalizeImage(imagePathString)
    // TODO: what is idiomatic way to do try-with in Scala?
    try {
      // Most important line - if normalization happens in GeoTrellis, adapt image parameter
      val labelProbabilities: Array[Float] = executeInceptionGraph(graphDef, image)

      // Task: Use thresholds to do multi-label classification
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
          // CSV format
          print(f"$label%s ")
          // LabelImage examplee format
          // println(f"MATCH: $label%s ($labelProbability%.2f%% likely)")
          // Table format
          // println(f"MATCH: $label%-20s ($i%d) $labelProbability%15.2f%% likely $threshold%15.2f%% threshold")
        }
      }
    } finally {
      image.close
    }
  }

  // If normalization happens in GeoTrellis, remove
  private def constructAndExecuteGraphToNormalizeImage = true
  def constructAndExecuteGraphToNormalizeImage(imagePathString: String): Tensor = {
    var g: Graph = null

    try {
      g = new Graph
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
      var imageTensor: Tensor = null
      var meansTensor: Tensor = null
      var stdsTensor: Tensor = null
      try {
        imageTensor = b.decodeTiff(imagePathString)

        val input: Output = b.constantTensor("input", imageTensor)

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

        meansTensor = Tensor.create(meansArray)
        stdsTensor = Tensor.create(stdsArray)
        val meansOutput: Output = b.constantTensor("means", meansTensor)
        val stdsOutput: Output = b.constantTensor("stds", stdsTensor)

        val output: Output =
          b.div(
            b.sub(
              b.expandDims(
                b.cast(input, DataType.FLOAT),
                b.constant("make_batch", 0)),
              meansOutput),
            stdsOutput)

          var s: Session = null
          try {
            s = new Session(g)
            return s.runner.fetch(output.op.name).run.get(0)
          } finally {
            s.close
          }
      } finally {
        imageTensor.close
        meansTensor.close
        stdsTensor.close
      }
    } finally {
      g.close
    }
  }

  private def executeInceptionGraph = true
  def executeInceptionGraph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    var g: Graph = null
    try {
      g = new Graph
      g.importGraphDef(graphDef)
      var s: Session = null
      var result: Tensor = null
      try {
        s = new Session(g)
        // Output layer, result shape, etc. will be different for non-tagging models
        result = s.runner.feed("input_1", image).fetch("dense/Sigmoid").run.get(0)
        val rshape: Array[Long] = result.shape
        val rshapeString: String = Arrays.toString(rshape)
        if (result.numDimensions != 2 || rshape(0) != 1) {
          throw new RuntimeException(
            String.format(
              f"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape $rshapeString%s"))
        }
        val nlabels: Int = rshape(1).asInstanceOf[Int]
        return result.copyTo(Array.ofDim[Float](1, nlabels))(0)
      } finally {
        s.close
      }
    } finally {
      g.close
    }
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
    return null
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
    return null
  }

  // In the fullness of time, equivalents of the methods of this class should be auto-generated from
  // the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
  // like Python, C++ and Go.
  class GraphBuilder(g: Graph) {
    def div(x: Output, y: Output): Output = {
      return binaryOp("Div", x, y)
    }

    def sub(x: Output, y: Output): Output = {
      return binaryOp("Sub", x, y)
    }

    def resizeBilinear(images: Output, size: Output): Output = {
      return binaryOp("ResizeBilinear", images, size)
    }

    def expandDims(input: Output, dim: Output): Output = {
      return binaryOp("ExpandDims", input, dim)
    }

    def cast(value: Output, dtype: DataType): Output = {
      return g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build.output(0)
    }

    // def decodeJpeg(contents: Output, channels: Long): Tensor = {
    //   // call decodeMultibandTile, which contains shared tensor logic
    //   return g.opBuilder("DecodeJpeg", "DecodeJpeg")
    //     .addInput(contents)
    //     .setAttr("channels", channels)
    //     .build
    //     .output(0)
    // }

    /**
     * Decode a TIFF-encoded image to a uint8 tensor using GeoTrellis.
     * TensorFlow Images: https://www.tensorflow.org/api_guides/python/image.
     * DecodeJpeg: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/decode-jpeg.
     */
    def decodeTiff(imagePathString: String): Tensor = {
      // Read GeoTiff: https://geotrellis.readthedocs.io/en/latest/tutorials/reading-geoTiffs.html
      val tile: MultibandTile = GeoTiffReader.readMultiband(imagePathString).tile

      val dataType: DataType = DataType.UINT8
      val height: Int = tile.rows
      val width: Int = tile.cols
      // TODO: HANDLE 4 channels!!
      val channels: Int = 3 //tile.bandCount
      val shape: Array[Long] = Array(height.asInstanceOf[Long], width.asInstanceOf[Long], channels.asInstanceOf[Long])
      val byteArray: Array[Byte] = new Array(height * width * channels)

      var h: Int = 0
      var w: Int = 0
      var c: Int = 0
      for (h <- 0 to height - 1) {
        for (w <- 0 to width - 1) {
          for (c <- 0 to channels - 1) {
            byteArray(h * (width * channels) + w * channels + c) = (tile.band(c).get(w, h)).asInstanceOf[Byte]
          }
        }
      }

      val data: ByteBuffer = ByteBuffer.wrap(byteArray)
      val imageTensor: Tensor = Tensor.create(dataType, shape, data)
      return imageTensor
    }

    def constant(name: String, value: Any): Output = {
      val t: Tensor = Tensor.create(value)
      val o: Output = g.opBuilder("Const", name)
        .setAttr("dtype", t.dataType)
        .setAttr("value", t)
        .build
        .output(0)
      t.close
      return o
    }

    def constantTensor(name: String, t: Tensor): Output = {
      return g.opBuilder("Const", name)
        .setAttr("dtype", t.dataType)
        .setAttr("value", t)
        .build
        .output(0)
    }

    private def binaryOp = true
    def binaryOp(ty: String, in1: Output, in2: Output): Output = {
      return g.opBuilder(ty, ty).addInput(in1).addInput(in2).build.output(0)
    }
  }
}
