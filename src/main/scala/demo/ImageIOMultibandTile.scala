package whocares

import geotrellis.raster._
import geotrellis.raster.io.geotiff._
import geotrellis.proj4._
import geotrellis.vector._
import com.twelvemonkeys.image.ResampleOp
import com.twelvemonkeys.imageio.AbstractMetadata
import com.twelvemonkeys.imageio.AbstractMetadata
import spire.syntax.cfor._

import java.awt.image._
import javax.imageio.ImageIO


object Main {
  def main(args: Array[String]): Unit = {
    // val readers = ImageIO.getImageReadersByFormatName("JPEG")
    // while (readers.hasNext()) {
    //   System.out.println("reader: " + readers.next())
    // }
    val img =
      ImageIO.read(new java.io.File("/Users/yoninachmany/azavea/raster-vision-data/datasets/planet_kaggle/train-jpg/train_1.jpg"));
    val mbt: MultibandTile = convertToMultibandTile(img)
    println(mbt)
    println(mbt.bands)
    println(mbt.toGeoTiffTile.segmentLayout)
    // issue: we return PixelInterleaveBandArrayTile, other way returns UByteRawArrayTile
    val m: MultibandGeoTiff = MultibandGeoTiff(mbt, Extent(0,0,256,256), LatLng, GeoTiffOptions.DEFAULT.copy(colorSpace = 2))
    // val m2: MultibandGeoTiff = new MultibandGeoTiff(m.tile.toGeoTiffTile.toArrayTile)
    m.write("/Users/yoninachmany/azavea/raster-vision-data/datasets/planet_kaggle/train-jpg/train_1.tif")
  }

  def convertToMultibandTile(image: BufferedImage): MultibandTile = {
    val numBands = image.getSampleModel.getNumBands
    val tiles = (0 until numBands).map({ i => convertToTile(image, i) })
    ArrayMultibandTile(tiles)
  }

  def convertToTile(renderedImage: BufferedImage, bandIndex: Int): Tile = {
    val buffer = renderedImage.getData.getDataBuffer
    val sampleModel = renderedImage.getSampleModel
    val rows: Int = renderedImage.getHeight
    val cols: Int = renderedImage.getWidth
    val cellType: CellType = getCellType(renderedImage)
    // println(cellType)

    def default =
      cellType match {
        case ct: DoubleCells =>
          val data = Array.ofDim[Double](cols * rows)
          sampleModel.getSamples(0, 0, cols, rows, bandIndex, data, buffer)
          DoubleArrayTile(data, cols, rows).convert(cellType)
        case ct: FloatCells =>
          val data = Array.ofDim[Float](cols * rows)
          sampleModel.getSamples(0, 0, cols, rows, bandIndex, data, buffer)
          FloatArrayTile(data, cols, rows).convert(cellType)
        case _ =>
          val data = Array.ofDim[Int](cols * rows)
          sampleModel.getSamples(0, 0, cols, rows, bandIndex, data, buffer)
          IntArrayTile(data, cols, rows).convert(cellType)
      }

    // println(buffer.getClass)
    buffer match {
      case byteBuffer: DataBufferByte =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = byteBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands

            cellType match {
              case ct: ByteCells =>
                PixelInterleaveBandArrayTile(ByteArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case ct: UByteCells =>
                val pixint = PixelInterleaveBandArrayTile(UByteArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
                val newData = pixint.toBytes
                val newInnerCols = pixint.cols
                val newRows = pixint.rows
                UByteArrayTile(newData, newInnerCols, newRows, ct)
                // PixelInterleaveBandArrayTile(UByteArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(UByteArrayTile(data, innerCols, rows, UByteCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = byteBuffer.getData(bandIndex)
            cellType match {
              case ct: ByteCells =>
                ByteArrayTile(data, cols, rows, ct)
              case ct: UByteCells =>
                UByteArrayTile(data, cols, rows, ct)
              case _ =>
                UByteArrayTile(data, cols, rows, UByteCellType).convert(cellType).toArrayTile
            }
          case mp: MultiPixelPackedSampleModel =>
            ???
            // // Tricky sample model, just do the slow direct thing.
            // val result = ArrayTile.alloc(cellType, cols, rows)
            // val values = Array.ofDim[Int](1)
            // cfor(0)(_ < cols, _ + 1) { col =>
            //   cfor(0)(_ < rows, _ + 1) { row =>
            //     gridCoverage.evaluate(new GridCoordinates2D(col, row), values)
            //     result.set(col, row, values(0))
            //   }
            // }
            // result
          case x =>
            default
        }
      case ushortBuffer: DataBufferUShort =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = ushortBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands
            cellType match {
              case ct: UShortCells =>
                PixelInterleaveBandArrayTile(UShortArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(UShortArrayTile(data, innerCols, rows, UShortCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = ushortBuffer.getData(bandIndex)
            cellType match {
              case ct: UShortCells =>
                UShortArrayTile(data, cols, rows, ct)
              case _ =>
                UShortArrayTile(data, cols, rows, UShortCellType).convert(cellType).toArrayTile
            }

          case _ =>
            default
        }
      case shortBuffer: DataBufferShort =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = shortBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands
            cellType match {
              case ct: ShortCells =>
                PixelInterleaveBandArrayTile(ShortArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(ShortArrayTile(data, innerCols, rows, ShortCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = shortBuffer.getData(bandIndex)
            cellType match {
              case ct: ShortCells =>
                ShortArrayTile(data, cols, rows, ct)
              case _ =>
                ShortArrayTile(data, cols, rows, ShortCellType).convert(cellType).toArrayTile
            }

          case _ =>
            default
        }
      case intBuffer: DataBufferInt =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = intBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands
            cellType match {
              case ct: IntCells =>
                PixelInterleaveBandArrayTile(IntArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case ct: FloatCells =>
                // Deal with unsigned ints
                val floatData = data.map { z => (z & 0xFFFFFFFFL).toFloat }
                PixelInterleaveBandArrayTile(FloatArrayTile(floatData, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(IntArrayTile(data, innerCols, rows, IntCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = intBuffer.getData(bandIndex)
            cellType match {
              case ct: IntCells =>
                IntArrayTile(data, cols, rows, ct)
              case ct: FloatCells =>
                // Deal with unsigned ints
                val floatData = data.map { z => (z & 0xFFFFFFFFL).toFloat }
                FloatArrayTile(floatData, cols, rows, ct)
              case _ =>
                IntArrayTile(data, cols, rows, IntCellType).convert(cellType).toArrayTile
            }

          case _ =>
            default
        }
      case floatBuffer: DataBufferFloat =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = floatBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands
            cellType match {
              case ct: FloatCells =>
                PixelInterleaveBandArrayTile(FloatArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(FloatArrayTile(data, innerCols, rows, FloatCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = floatBuffer.getData(bandIndex)
            cellType match {
              case ct: FloatCells =>
                FloatArrayTile(data, cols, rows, ct)
              case _ =>
                FloatArrayTile(data, cols, rows, FloatCellType).convert(cellType).toArrayTile
            }

          case _ =>
            default
        }
      case doubleBuffer: DataBufferDouble =>
        sampleModel match {
          case _: PixelInterleavedSampleModel =>
            val data = doubleBuffer.getData()
            val numBands = sampleModel.getNumBands
            val innerCols = cols * numBands
            cellType match {
              case ct: DoubleCells =>
                PixelInterleaveBandArrayTile(DoubleArrayTile(data, innerCols, rows, ct), numBands, bandIndex)
              case _ =>
                PixelInterleaveBandArrayTile(DoubleArrayTile(data, innerCols, rows, DoubleCellType).convert(cellType).toArrayTile, numBands, bandIndex)
            }

          case _: BandedSampleModel =>
            val data = doubleBuffer.getData(bandIndex)
            cellType match {
              case ct: DoubleCells =>
                DoubleArrayTile(data, cols, rows, ct)
              case _ =>
                DoubleArrayTile(data, cols, rows, DoubleCellType).convert(cellType).toArrayTile
            }

          case _ =>
            default
        }
    }
  }

  def getCellType(renderedImage: BufferedImage): CellType =
      // UShortCellType
    UByteCellType
  //   val buffer = renderedImage.getData.getDataBuffer
  //   val dataType = buffer.getDataType
  //   val sampleType = renderedImage.getSampleModel.getSampleDimension(0).getSampleDimensionType
  //   val noDataValue: Option[Double] = None
  //   (noDataValue, dataType, sampleType) match {

  //     // Bit

  //     case (_, DataBuffer.TYPE_BYTE, SampleDimensionType.UNSIGNED_1BIT) =>
  //       BitCellType

  //     // Unsigned Byte

  //     case (Some(nd), DataBuffer.TYPE_BYTE, SampleDimensionType.UNSIGNED_8BITS) if (nd.toInt > 0 && nd <= 255) =>
  //       UByteUserDefinedNoDataCellType(nd.toByte)
  //     case (Some(nd), DataBuffer.TYPE_BYTE, SampleDimensionType.UNSIGNED_8BITS) if nd.toByte == ubyteNODATA =>
  //       UByteConstantNoDataCellType
  //     case (_, DataBuffer.TYPE_BYTE, SampleDimensionType.UNSIGNED_8BITS) =>
  //       UByteCellType

  //     // Signed Byte

  //     case (Some(nd), DataBuffer.TYPE_BYTE, _) if (nd.toInt > Byte.MinValue.toInt && nd <= Byte.MaxValue.toInt) =>
  //       ByteUserDefinedNoDataCellType(nd.toByte)
  //     case (Some(nd), DataBuffer.TYPE_BYTE, _) if (nd.toByte == byteNODATA) =>
  //       ByteConstantNoDataCellType
  //     case (_, DataBuffer.TYPE_BYTE, _) =>
  //       ByteCellType

  //     // Unsigned Short

  //     case (Some(nd), DataBuffer.TYPE_USHORT, _) if (nd.toInt > 0 && nd <= 65535) =>
  //       UShortUserDefinedNoDataCellType(nd.toShort)
  //     case (Some(nd), DataBuffer.TYPE_USHORT, _) if (nd.toShort == ushortNODATA) =>
  //       UShortConstantNoDataCellType
  //     case (_, DataBuffer.TYPE_USHORT, _) =>
  //       UShortCellType

  //     // Signed Short

  //     case (Some(nd), DataBuffer.TYPE_SHORT, _) if (nd > Short.MinValue.toDouble && nd <= Short.MaxValue.toDouble) =>
  //       ShortUserDefinedNoDataCellType(nd.toShort)
  //     case (Some(nd), DataBuffer.TYPE_SHORT, _) if (nd.toShort == shortNODATA) =>
  //       ShortConstantNoDataCellType
  //     case (_, DataBuffer.TYPE_SHORT, _) =>
  //       ShortCellType


  //     // Unsigned Int32

  //     case (Some(nd), DataBuffer.TYPE_INT, SampleDimensionType.UNSIGNED_32BITS) if (nd.toLong > 0L && nd.toLong <= 4294967295L) =>
  //       FloatUserDefinedNoDataCellType(nd.toFloat)
  //     case (Some(nd), DataBuffer.TYPE_INT, SampleDimensionType.UNSIGNED_32BITS) if (nd.toLong == 0L) =>
  //       FloatConstantNoDataCellType
  //     case (None, DataBuffer.TYPE_INT, SampleDimensionType.UNSIGNED_32BITS) =>
  //       FloatCellType

  //     // Signed Int32

  //     case (Some(nd), DataBuffer.TYPE_INT, _) if (nd.toInt > Int.MinValue && nd.toInt <= Int.MaxValue) =>
  //       IntUserDefinedNoDataCellType(nd.toInt)
  //     case (Some(nd), DataBuffer.TYPE_INT, _) if (nd.toInt == NODATA) =>
  //       IntConstantNoDataCellType
  //     case (_, DataBuffer.TYPE_INT, _) =>
  //       IntCellType

  //     // Float

  //     case (Some(nd), DataBuffer.TYPE_FLOAT, _) if (isData(nd) & Float.MinValue.toDouble <= nd & Float.MaxValue.toDouble >= nd) =>
  //       FloatUserDefinedNoDataCellType(nd.toFloat)
  //     case (Some(nd), DataBuffer.TYPE_FLOAT, _) =>
  //       FloatConstantNoDataCellType
  //     case (None, DataBuffer.TYPE_FLOAT, _) =>
  //       FloatCellType

  //     // Double

  //     case (Some(nd), DataBuffer.TYPE_DOUBLE, _) if isData(nd) =>
  //       DoubleUserDefinedNoDataCellType(nd)
  //     case (Some(nd), DataBuffer.TYPE_DOUBLE, _) =>
  //       DoubleConstantNoDataCellType
  //     case (None, DataBuffer.TYPE_DOUBLE, _) =>
  //       DoubleCellType

  //     case _ =>
  //       throw new Exception(s"Unable to convert GridCoverage2D: Unsupported CellType (NoData=${noDataValue}  TypeEnum=${dataType}  SampleDimensionType=${sampleType}")
  //   }
  // }

}
