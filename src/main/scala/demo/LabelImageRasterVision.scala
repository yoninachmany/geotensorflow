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

import org.tensorflow.{Tensor, TensorFlow}

import java.io.PrintStream
import java.nio.file.Paths
import java.util.List

/** Sample use of the TensorFlow Java API to label images using a pre-trained model. */
object LabelImageRasterVision {
  def printUsage(s: PrintStream) {
    s.println(
      "Scala program that uses a pre-trained Raster Vision model (https://github.com/azavea/raster-vision)")
    s.println("to label Jpeg images.")
    s.println("TensorFlow version: " + TensorFlow.version)
    s.println
    s.println("Usage: label_image <run name> <image file>")
    s.println
    s.println("Where:")
    s.println("<run name> is the unique ID for the Raster Vision experiment")
    s.println("<image file> is the path to a JPEG image file")
  }

  def main(args: Array[String]) {
    if (args.length != 2) {
      printUsage(System.err)
      System.exit(1)
    }
    val runName: String = args(0)
    val imageFile: String = args(1)

    val graphDef: Array[Byte] = LabelImageUtils.readAllBytesOrExit(RasterVisionUtils.getGraphPath(runName))
    val labels: List[String] = LabelImageUtils.readAllLinesOrExit(Paths.get("planet_kaggle_label_strings.txt"))
    val imagePathString: String = Paths.get(imageFile).toString

    var image: Tensor = RasterVisionUtils.constructAndExecuteGraphToNormalizeRasterVisionImage(imagePathString)
    try {
      val labelProbabilities: Array[Float] = RasterVisionUtils.executeRasterVisionTaggingGraph(graphDef, image)
      RasterVisionUtils.printMatches(runName, labelProbabilities, labels)
    } finally {
      image.close
    }
  }
}
