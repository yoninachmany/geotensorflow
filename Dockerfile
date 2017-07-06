FROM tensorflow/tensorflow

MAINTAINER Yoni Nachmany, yoninachmany@gmail.com

# Java: https://wiki.debian.org/Java
RUN apt-get update
RUN apt-get install default-jre -y
RUN apt-get install default-jdk -y
