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
object LabelImageInception {
  def printUsage(s: PrintStream) {
    val url: String =
      "https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip"
    s.println(
      "Scala program that uses a pre-trained Inception model (http://arxiv.org/abs/1512.00567)")
    s.println("to label JPEG images.")
    s.println("TensorFlow version: " + TensorFlow.version)
    s.println
    s.println("Usage: label_image <model dir> <image file>")
    s.println
    s.println("Where:")
    s.println("<model dir> is a directory containing the unzipped contents of the inception model")
    s.println("            (from " + url + ")")
    s.println("<image file> is the path to a JPEG image file")
  }

  def main(args: Array[String]) {
    if (args.length != 2) {
      printUsage(System.err)
      System.exit(1)
    }
    val modelDir: String = args(0)
    val imageFile: String = args(1)

    val graphDef: Array[Byte] = readAllBytesOrExit(Paths.get(modelDir, "tensorflow_inception_graph.pb"))
    val labels: List[String] = readAllLinesOrExit(Paths.get(modelDir, "imagenet_comp_graph_label_strings.txt"))
    val imageBytes: Array[Byte] = readAllBytesOrExit(Paths.get(imageFile));
    // val imagePathString: String = Paths.get(imageFile).toString

    var image: Tensor = constructAndExecuteGraphToNormalizeImage(imageBytes)
    try {
      val labelProbabilities: Array[Float] = executeInceptionGraph(graphDef, image)
      val bestLabelIdx: Int = maxIndex(labelProbabilities)
      val bestLabel: String = labels.get(bestLabelIdx)
      val bestLabelLikelihood: Float = labelProbabilities(bestLabelIdx) * 100f
      println(f"BEST MATCH: $bestLabel%s ($bestLabelLikelihood%.2f%% likely)")
    } finally {
      image.close
    }
  }

  private def constructAndExecuteGraphToNormalizeImage = true
  def constructAndExecuteGraphToNormalizeImage(imageBytes: Array[Byte]): Tensor = {
    var g: Graph = null

    try {
      g = new Graph
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
      var s: Session = null
      try {
        s = new Session(g)
        return s.runner.fetch(output.op.name).run.get(0)
      } finally {
        s.close
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
        result = s.runner().feed("input", image).fetch("output").run().get(0)
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

  private def maxIndex = true
  def maxIndex(probabilities: Array[Float]): Int = {
    var best: Int = 0
    val i: Int = 1
    for (i <- 1 to probabilities.length-1) {
      if (probabilities(i) > probabilities(best)) {
        best = i
      }
    }
    return best
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

    def decodeJpeg(contents: Output, channels: Long): Output = {
      return g.opBuilder("DecodeJpeg", "DecodeJpeg")
          .addInput(contents)
          .setAttr("channels", channels)
          .build
          .output(0);
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
