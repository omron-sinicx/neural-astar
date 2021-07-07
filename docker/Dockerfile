FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-devel

RUN apt-get -y update && apt-get -y install --no-install-recommends wget parallel
RUN pip install -U pip 

WORKDIR /workspace

ENV SHELL /bin/bash

COPY setup.py .
RUN pip install -e . && rm setup.py
COPY thirdparty/segmentation_models_pytorch thirdparty/segmentation_models_pytorch 
# make sure it installs segmentation_models_pytorch version 0.1.0
RUN pip install thirdparty/segmentation_models_pytorch && rm -r thirdparty/segmentation_models_pytorch  
