FROM tensorflow/tensorflow

MAINTAINER Yoni Nachmany, yoninachmany@gmail.com

# Java: https://wiki.debian.org/Java
RUN apt-get update
RUN apt-get install default-jre -y
RUN apt-get install default-jdk -y

COPY target/scala-2.11/geotrellis-sbt-template-assembly-0.1.0.jar /tmp
