FROM pytorch/pytorch:1.8.1-cuda10.2-cudnn7-devel

WORKDIR /workspace

RUN pip install -U pip setuptools

WORKDIR /workspace
COPY pyproject.toml .
RUN pip install -e .[dev]
