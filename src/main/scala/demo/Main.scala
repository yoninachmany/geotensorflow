package demo

import geotrellis.raster._
import geotrellis.spark._
import LabelImage._

object Main {
  def helloSentence = "Hello GeoTrellis"

  def main(args: Array[String]): Unit = {
    // Size is 438, 406 for 534 KB -> ~700 ms
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1002.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1051.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1078.tif"))
    LabelImage.main(Array("inception5h", "data/3band_AOI_1_RIO_img1088.tif"))
    // Size is 650, 650 for 2.5 MB -> ~1100 ms
    LabelImage.main(Array("inception5h", "data/RGB-PanSharpen_AOI_3_Paris_img1019.tif"))
    LabelImage.main(Array("inception5h", "data/RGB-PanSharpen_AOI_3_Paris_img1089.tif"))
    LabelImage.main(Array("inception5h", "data/RGB-PanSharpen_AOI_3_Paris_img1136.tif"))
    // Exception: compression type JPEG is not supported by this reader.
    // LabelImage.main(Array("inception5h", "data/013022223103.tif"))
    // LabelImage.main(Array("inception5h", "data/013022223121.tif"))
  }
}
