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
  private def executePreTrainedGraph = true
  def executePreTrainedGraph(graphDef: Array[Byte], image: Tensor, inputOp: String, outputOp: String): Array[Float] = {
    var g: Graph = new Graph
    g.importGraphDef(graphDef)
    var s: Session = new Session(g)
    var result: Tensor = s.runner().feed(inputOp, image).fetch(outputOp).run().get(0)
    val rshape: Array[Long] = result.shape
    val rshapeString: String = Arrays.toString(rshape)
    if (result.numDimensions != 2 || rshape(0) != 1) {
      throw new RuntimeException(
        String.format(
          f"Expected model to produce a [1 N] shaped tensor where N is the number of labels, instead it produced one with shape $rshapeString%s"))
    }
    val nlabels: Int = rshape(1).asInstanceOf[Int]
    // g.close
    // s.close
    result.copyTo(Array.ofDim[Float](1, nlabels))(0)
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
}
