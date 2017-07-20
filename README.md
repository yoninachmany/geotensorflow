GeoTensorFlow
=======================

## Goal: label training image from [Planet Kaggle](https://github.com/azavea/raster-vision#planet-kaggle)

![Kaggle image](train_1.jpg)

> agriculture clear primary water

## Quick Start

```console
git clone https://github.com/yoninachmany/geotensorflow.git
cd geotensorflow
(./inception5h/download.sh)
sbt "run-main demo.LabelImageInception inception5h train_1.tif"
```

> BEST MATCH: corn (3.43% likely)

## Work with Raster Vision

Follow the [Raster Vision](https://github.com/azavea/raster-vision) instructions to setup and run experiments locally.

```console
sbt "run-main demo.LabelImageRasterVision <run_name> train_1.tif"
```

> Sample output:
> MATCH: agriculture (93.62% likely)
> MATCH: artisinal_mine (53.25% likely)
> MATCH: bare_ground (80.82% likely)
> MATCH: blow_down (45.81% likely)
> MATCH: clear (83.24% likely)
> MATCH: cloudy (63.95% likely)
> MATCH: cultivation (40.01% likely)
> MATCH: habitation (97.28% likely)
> MATCH: haze (34.41% likely)
> MATCH: partly_cloudy (49.84% likely)
> MATCH: primary (87.78% likely)
> MATCH: road (60.91% likely)
