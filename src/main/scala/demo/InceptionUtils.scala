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
  def constructAndExecuteGraphToNormalizeInceptionV5Image(imageBytes: Array[Byte]): Tensor = {
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

    val s: Session = new Session(g)
    return s.runner.fetch(output.op.name).run.get(0)
    val result: Tensor = s.runner.fetch(output.op.name).run.get(0)
    result
    // g.close
    // s.close
  }

  def constructAndExecuteGraphToNormalizeInceptionV3Image(imageBytes: Array[Byte]): Tensor = {
    var g: Graph = new Graph
    val b: GraphBuilder = new GraphBuilder(g)

    // Some constants specific to the pre-trained model at:
    // https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip
    //
    // - The model was trained with images scaled to 224x224 pixels.
    // - The colors, represented as R, G, B in 1-byte each were converted to
    //   float using (value - Mean)/Scale.
    // https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/label_image/main.cc#L285-L288
    val H: Int = 299
    val W: Int = 299
    val mean: Float = 0
    val scale: Float = 255

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

  def executeInceptionV5Graph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    LabelImageUtils.executePreTrainedGraph(graphDef, image, "input", "output")
  }

  def executeInceptionV3Graph(graphDef: Array[Byte], image: Tensor): Array[Float] = {
    LabelImageUtils.executePreTrainedGraph(graphDef, image, "input_1", "predictions/Softmax")
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
    val bestLabelIdx: Int = maxIndex(labelProbabilities) //93//169//4
    val bestLabel: String = labels.get(bestLabelIdx)
    val bestLabelLikelihood: Float = labelProbabilities(bestLabelIdx) * 100f
    println(f"BEST MATCH: $bestLabel%s ($bestLabelLikelihood%.2f%% likely)")
  }
}
