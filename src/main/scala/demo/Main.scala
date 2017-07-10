package demo

import geotrellis.raster._
import geotrellis.spark._
import LabelImage._

object Main {
  def helloSentence = "Hello GeoTrellis"

  def main(args: Array[String]): Unit = {
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1002.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1051.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1078.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1088.tif"))
  }
}
