FROM nvidia/cuda:11.6.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y update && apt-get -y install --no-install-recommends software-properties-common libgl1-mesa-dev wget libssl-dev

RUN apt-get -y install --no-install-recommends python3.8-dev python3.8-distutils python3-pip python3.8-venv
# Set default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1

# clear cache
RUN rm -rf /var/lib/apt/lists/*

RUN pip3 install -U pip distlib setuptools wheel

WORKDIR /workspace

WORKDIR /workspace
COPY src/ src/
COPY pyproject.toml .
RUN pip3 install -e .[dev]
RUN pip3 uninstall -y torch torchvision
RUN pip3 install torch==1.12.1 torchvision==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
