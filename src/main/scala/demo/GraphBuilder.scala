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

import geotrellis.raster._
import geotrellis.raster.{Tile, MultibandTile}
import geotrellis.raster.io.geotiff.reader.GeoTiffReader
import org.tensorflow.{DataType, Graph, Output, Tensor}
import spray.json._
import DefaultJsonProtocol._

import java.nio.ByteBuffer
import java.nio.file.Paths
import java.awt.image.BufferedImage
import javax.imageio.ImageIO
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

  def castFloat(value: Output): Output = {
    g.opBuilder("Cast", "CastFloat").addInput(value).setAttr("DstT", DataType.FLOAT).build.output(0)
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

  private def getMultibandTileFromJpeg = true
  def getMultibandTileFromJpeg(imagePathString: String): MultibandTile = {
    val image: BufferedImage = ImageIO.read(new java.io.File(imagePathString))
    val mbt = ImageIOMultibandTile.convertToMultibandTile(image)
    import geotrellis.raster.io.geotiff._
    import geotrellis.vector._
    import geotrellis.proj4._
    GeoTiff(mbt, Extent(0,0,1,1), LatLng).write("reversed.tiff")
    mbt
  }

  private def decodeMultibandTile = true
  def decodeMultibandTile(tile: MultibandTile): Tensor = {
    val height: Int = tile.rows
    val width: Int = tile.cols
    // TODO: HANDLE 4 channels!!
    val channels: Int = 3 //tile.bandCount
    val imageData: Array[Array[Array[Int]]] = Array.ofDim[Int](height, width, channels)

    var h: Int = 0
    var w: Int = 0
    var c: Int = 0
    for (h <- 0 to height - 1) {
      for (w <- 0 to width - 1) {
        for (c <- 0 to channels - 1) {
          imageData(h)(w)(c) = tile.band(c).get(w, h)
        }
      }
    }
    val imageTensor: Tensor = Tensor.create(imageData)
    imageTensor
  }

  /**
   * Decode a JPEG-encoded image to a uint8 tensor with a GeoTrellis MultibandTile.
   * DecodeJpeg: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/decode-jpeg.
   */
  def decodeJpegGeoTrellis(imagePathString: String): Tensor = {
    val tile: MultibandTile = getMultibandTileFromJpeg(imagePathString)
    decodeMultibandTile(tile)
  }

  private def normalizeMultibandTile = true
  def normalizeMultibandTile(tile: MultibandTile): MultibandTile = {
    val stats: Map[String, Array[Double]] = RasterVisionUtils.readChannelStats
    val means: Array[Double] = stats("means")
    val stds: Array[Double] = stats("stds")

    val normalized: MultibandTile =
      tile.mapBands { (bandIndex, band) =>
        (band.convert(DoubleConstantNoDataCellType) - means(bandIndex)) / stds(bandIndex)
      }
    normalized
  }

  private def normalizeMultibandTileForInception = true
  def normalizeMultibandTileForInception(tile: MultibandTile, mean: Float, scale: Float): MultibandTile = {
    val normalized: MultibandTile =
      tile.mapBands { (bandIndex, band) =>
        (band.convert(DoubleConstantNoDataCellType) - mean) / scale
      }
    normalized
  }

  /**
   * Decode and normalize a JPEG-encoded image to a uint8 tensor with a GeoTrellis MultibandTile.
   * DecodeJpeg: https://www.tensorflow.org/api_docs/cc/class/tensorflow/ops/decode-jpeg.
   */
  def decodeAndNormalizeJpegGeoTrellis(imagePathString: String): Tensor = {
    var tile: MultibandTile = getMultibandTileFromJpeg(imagePathString)
    val normalized: MultibandTile = normalizeMultibandTile(tile)
    decodeMultibandTile(normalized)
  }

  def decodeAndNormalizeJpegGeoTrellisForInception(imagePathString: String, mean: Float, scale: Float): Tensor = {
    var tile: MultibandTile = getMultibandTileFromJpeg(imagePathString)
    val normalized: MultibandTile = normalizeMultibandTileForInception(tile, mean, scale)
    decodeMultibandTile(normalized)
  }


}
