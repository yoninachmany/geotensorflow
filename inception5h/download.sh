#!/bin/sh
curl https://storage.googleapis.com/download.tensorflow.org/models/inception5h.zip > inception5h.zip &&
unzip inception5h.zip && rm inception5h.zip
