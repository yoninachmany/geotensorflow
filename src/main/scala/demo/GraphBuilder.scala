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

import org.tensorflow.{DataType, Graph, Output, Tensor}

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

  private def binaryOp = true
  def binaryOp(ty: String, in1: Output, in2: Output): Output = {
    return g.opBuilder(ty, ty).addInput(in1).addInput(in2).build.output(0)
  }

  def constantTensor(name: String, t: Tensor): Output = {
    return g.opBuilder("Const", name)
      .setAttr("dtype", t.dataType)
      .setAttr("value", t)
      .build
      .output(0)
  }
}
