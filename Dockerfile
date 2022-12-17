FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

WORKDIR /workspace

RUN pip install -U pip setuptools

WORKDIR /workspace
COPY src/ src/
COPY pyproject.toml .
RUN pip install -e .[dev]
