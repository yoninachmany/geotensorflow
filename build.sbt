// Rename this as you see fit
name := "geotrellis-sbt-template"

version := "0.1.0"

scalaVersion := "2.11.8"

organization := "com.azavea"

licenses := Seq("Apache-2.0" -> url("http://www.apache.org/licenses/LICENSE-2.0.html"))

scalacOptions ++= Seq(
  "-deprecation",
  "-unchecked",
  "-Yinline-warnings",
  "-language:implicitConversions",
  "-language:reflectiveCalls",
  "-language:higherKinds",
  "-language:postfixOps",
  "-language:existentials",
  "-feature")

publishMavenStyle := true
publishArtifact in Test := false
pomIncludeRepository := { _ => false }

resolvers += Resolver.bintrayRepo("azavea", "geotrellis")

libraryDependencies ++= Seq(
  "com.azavea.geotrellis" %% "geotrellis-spark" % "1.0.0-d9f051d",
  "com.azavea.geotrellis" %% "geotrellis-raster"% "1.0.0-d9f051d",
  "org.apache.spark"      %% "spark-core"       % "2.0.1" % "provided",
  "org.scalatest"         %%  "scalatest"       % "2.2.0" % "test"
)

// https://mvnrepository.com/artifact/org.tensorflow/tensorflow
libraryDependencies += "org.tensorflow" % "tensorflow" % "1.2.1"

// When creating fat jar, remote some files with
// bad signatures and resolve conflicts by taking the first
// versions of shared packaged types.
assemblyMergeStrategy in assembly := {
  case "reference.conf" => MergeStrategy.concat
  case "application.conf" => MergeStrategy.concat
  case "META-INF/MANIFEST.MF" => MergeStrategy.discard
  case "META-INF\\MANIFEST.MF" => MergeStrategy.discard
  case "META-INF/ECLIPSEF.RSA" => MergeStrategy.discard
  case "META-INF/ECLIPSEF.SF" => MergeStrategy.discard
  case _ => MergeStrategy.first
}
