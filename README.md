GeoTrellis SBT TensorFlow
=======================

This is an application that uses GeoTrellis to read in GeoTiffs to TensorFlow.

An `sbt` bootstrapping script is also supplied, so you don't even need to
have `sbt` installed on your system.

To fetch this repo:

```console
git clone https://github.com/geotrellis/geotrellis-sbt-template.git
cd geotrellis-sbt-tensorflow
```

Make sure to unzip `inception5h.zip` in its own directory.

To build the assembly .jar file:

```console
sbt assembly
```

To run (example <model dir> is `inception5h` and <image file> is `sample.tif`):
```console
sbt
compile
run <model dir> <image file>
```
