#!/bin/sh
curl https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip > inception5h/inception5h.zip &&
unzip inception5h/inception5h.zip -d inception5h && rm inception5h/inception5h.zip
