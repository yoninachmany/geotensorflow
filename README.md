GeoTrellis SBT TensorFlow
=======================

This is an application that uses GeoTrellis to read in GeoTiffs to TensorFlow for image ML tasks.

### To fetch this repo:

```console
git clone https://github.com/geotrellis/geotrellis-sbt-template.git
cd geotrellis-sbt-tensorflow
```

### To build the assembly .jar file:

```console
sbt assembly
```

### To run:
```console
sbt
compile
run inception5h sample.tif
```

### Results:
**SpaceNetImage - originally tiff**
![SpaceNet image](spacenet.png)

BEST MATCH: fountain (8.91% likely)

**Example jpg**
![Example jpg](example-400x288.jpg)

(original jpg code and image) BEST MATCH: lakeside (19.00% likely)
(modified tif code and image) BEST MATCH: lakeside (18.52% likely)
