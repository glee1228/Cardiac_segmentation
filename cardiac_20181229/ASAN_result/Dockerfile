#!/bin/sh
FROM tensorflow/tensorflow:latest-gpu-py3
WORKDIR /


RUN pip install SimpleITK
RUN pip install numpy
RUN apt-get update && apt-get install -y git-core
RUN pip install git+https://www.github.com/keras-team/keras-contrib.git
RUN pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
RUN pip install scipy
RUN pip install keras
RUN pip install matplotlib
RUN nvcc --version
ENTRYPOINT /train.sh ; /bin/bash

COPY . .
