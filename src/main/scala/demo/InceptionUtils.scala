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

object InceptionUtils {
  def constructAndExecuteGraphToNormalizeInceptionImageFromPath(imagePathString: String, modelDir: String): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)

    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val isInception5h: Boolean = modelDir == "inception5h"
    val H: Int = if (isInception5h) 224 else 299
    val W: Int = if (isInception5h) 224 else 299
    val mean: Float = if (isInception5h) 117f else 0f
    val scale: Float = if (isInception5h) 1f else 255f

    // Since the graph is being constructed once per execution here, we can use a constant for the
    // input image. If the graph were to be re-used for multiple input images, a placeholder would
    // have been more appropriate.
    var imageTensor: Tensor = b.decodeAndNormalizeJpegGeoTrellisForInception(imagePathString, mean, scale)
    val input: Output = b.constantTensor("input", imageTensor)
    val output: Output =
      b.div(
        b.sub(
          b.resizeBilinear(
          b.expandDims(
            b.cast(input, DataType.FLOAT),
            b.constant("make_batch", 0)),
          b.constant("size", Array[Int](H, W))),
          b.constant("mean", mean)),
        b.constant("scale", scale))

    val s: Session = new Session(g)
    return s.runner.fetch(output.op.name).run.get(0)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    result
    // g.close
    // s.close
  }

  def constructAndExecuteGraphToNormalizeInceptionImage(imageBytes: Array[Byte], modelDir: String): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)

    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    val isInception5h: Boolean = modelDir == "inception5h"
    val H: Int = if (isInception5h) 224 else 299
    val W: Int = if (isInception5h) 224 else 299
    val mean: Float = if (isInception5h) 117f else 0f
    val scale: Float = if (isInception5h) 1f else 255f

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

    val s: Session = new Session(g)
    return s.runner.fetch(output.op.name).run.get(0)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    result
    // g.close
    // s.close
  }

  def executeInceptionGraph(graphDef: Array[Byte], image: Tensor, modelDir: String): Array[Float] = {
    val isInception5h: Boolean = modelDir == "inception5h"
    val wasProvided: Boolean = modelDir != "inception3-handmade"
    val inputOp: String = if (wasProvided) "input" else "input_2"
    val outputOp: String = if (isInception5h) "output" else (if (wasProvided) "InceptionV3/Predictions/Reshape_1" else "predictions/Softmax")
    LabelImageUtils.executePreTrainedGraph(graphDef, image, inputOp, outputOp)
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

  def printBestMatch(labelProbabilities: Array[Float], labels: List[String]) {
    val bestLabelIdx: Int = maxIndex(labelProbabilities)
    val bestLabel: String = labels.get(bestLabelIdx)
    val bestLabelLikelihood: Float = labelProbabilities(bestLabelIdx) * 100f
    println(f"BEST MATCH: $bestLabel%s ($bestLabelLikelihood%.2f%% likely)")
  }

  def getGraphPath(modelDir: String): Path = {
    val graphFilename: String = if (modelDir == "inception3") "inception_v3_2016_08_28_frozen.pb" else "tensorflow_inception_graph.pb"
    Paths.get(modelDir, graphFilename)
  }

  def getLabelsPath(modelDir: String): Path = {
    val labelsFilename: String = if (modelDir == "inception3") "imagenet_slim_labels.txt" else "imagenet_comp_graph_label_strings.txt"
    Paths.get(modelDir, labelsFilename)
  }
}
