FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

WORKDIR /workspace

RUN pip install -U pip

WORKDIR /workspace
COPY setup.py .
RUN pip install -e . && rm setup.py
