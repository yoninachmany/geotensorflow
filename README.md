GeoTrellis SBT TensorFlow
=======================

This is an application that uses GeoTrellis to read in GeoTiffs to TensorFlow.

To fetch this repo:

```console
git clone https://github.com/geotrellis/geotrellis-sbt-template.git
cd geotrellis-sbt-tensorflow
```

To build the assembly .jar file:

```console
sbt assembly
```

To run:
```console
sbt
compile
run inception5h sample.tif
```
