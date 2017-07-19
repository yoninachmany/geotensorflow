Raster Vision predictions run with GeoTrellis
=======================

## Quick Start

```console
git clone https://github.com/yoninachmany/geotensorflow.git
cd geotensorflow
(cd model; sh download.sh) # download inception
sbt "core/run-main demo.LabelImage spacenet.tif"
```


```

### Results:
**SpaceNetImage - originally tiff**

BEST MATCH: fountain (8.91% likely)

![SpaceNet image](spacenet.png)

**Example jpg**

(original jpg code and image) BEST MATCH: lakeside (19.00% likely)

(modified tif code and image) BEST MATCH: lakeside (18.52% likely)

![Example jpg](example-400x288.jpg)
