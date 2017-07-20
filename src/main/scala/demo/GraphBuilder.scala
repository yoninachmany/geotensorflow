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

import geotrellis.raster.{Tile, MultibandTile}
import geotrellis.raster.io.geotiff.MultibandGeoTiff
import geotrellis.raster.io.geotiff.reader.GeoTiffReader
import org.tensorflow.{DataType, Graph, Output, Tensor}

import java.nio.ByteBuffer
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
import ImageIOMultibandTile.convertToMultibandTile

// In the fullness of time, equivalents of the methods of this class should be auto-generated from
// the OpDefs linked into libtensorflow_jni.so. That would match what is done in other languages
// like Python, C++ and Go.
// https://github.com/tensorflow/tensorflow/blob/r1.2/tensorflow/java/src/main/java/org/tensorflow/examples/LabelImage.java#L156-L207
class GraphBuilder(g: Graph) {
  def div(x: Output, y: Output): Output = binaryOp("Div", x, y)
  def sub(x: Output, y: Output): Output = binaryOp("Sub", x, y)
  def resizeBilinear(images: Output, size: Output): Output = binaryOp("ResizeBilinear", images, size)
  def expandDims(input: Output, dim: Output): Output = binaryOp("ExpandDims", input, dim)

  def cast(value: Output, dtype: DataType): Output = {
    g.opBuilder("Cast", "Cast").addInput(value).setAttr("DstT", dtype).build.output(0)
  }

  def decodeJpeg(contents: Output, channels: Long): Output = {
    g.opBuilder("DecodeJpeg", "DecodeJpeg").addInput(contents).setAttr("channels", channels).build.output(0);
  }

  def constant(name: String, value: Any): Output = {
    val t: Tensor = Tensor.create(value)
    val o: Output = g.opBuilder("Const", name).setAttr("dtype", t.dataType).setAttr("value", t).build.output(0)
    t.close
    o
  }

  private def binaryOp = true
  def binaryOp(ty: String, in1: Output, in2: Output): Output = g.opBuilder(ty, ty).addInput(in1).addInput(in2).build.output(0)

  // Added
  def constantTensor(name: String, t: Tensor): Output = g.opBuilder("Const", name).setAttr("dtype", t.dataType).setAttr("value", t).build.output(0)

  private def isJpg = true
  def isJpg(imagePathString: String): Boolean = imagePathString.indexOfSlice("jpg") != -1 || imagePathString.indexOfSlice("jpg") != -1

  private def getMultibandTileFromJpg = true
  def getMultibandTileFromJpg(imagePathString: String): MultibandTile = {
    val image: BufferedImage = ImageIO.read(new java.io.File(imagePathString));
    ImageIOMultibandTile.convertToMultibandTile(image)
  }

  private def getMultibandTileFromTif = true
  def getMultibandTileFromTif(imagePathString: String): MultibandTile = GeoTiffReader.readMultiband(imagePathString).tile

  /**
   * Decode a TIFF-encoded image to a uint8 tensor using GeoTrellis.
   * TensorFlow Images: https://www.tensorflow.org/api_guides/python/image.
   * DecodeJpeg: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/decode-jpeg.
   */
  def decodeWithMultibandTile(imagePathString: String): Tensor = {
    // Read GeoTiff: https://geotrellis.readthedocs.io/en/latest/tutorials/reading-geoTiffs.html
    var tile: MultibandTile = if (isJpg(imagePathString)) then getMultibandTileFromJpg(imagePathString) else getMultibandTileFromTif(imagePathString)

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
    imageTensor
  }
}
