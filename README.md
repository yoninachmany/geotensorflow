Raster Vision predictions run with GeoTrellis
=======================

## Quick Start

```console
git clone https://github.com/yoninachmany/geotensorflow.git
cd geotensorflow
(./inception5h/download.sh)
sbt "run-main demo.LabelImage inception5h train_1.tif"
# BEST MATCH: corn (3.43% likely)
```

![Kaggle image](train_1.jpg)
