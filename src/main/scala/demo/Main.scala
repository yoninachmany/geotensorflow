package demo

import geotrellis.raster._
import geotrellis.spark._
import LabelImage._

object Main {
  def helloSentence = "Hello GeoTrellis"

  def main(args: Array[String]): Unit = {
    // println(helloSentence)
    LabelImage.main(Array("inception5h", "data/013022223121.tif"))
  }
}
