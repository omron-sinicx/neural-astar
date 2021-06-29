#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name="neural_astar",
    version="0.1.0",
    description="Path Planning using Neural A* Search",
    author="Ryo Yonetani",
    author_email="ryo.yonetani@sinicx.com",
    url="https://github.com/omron-sinicx/neural-astar",
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
		"segmentation-models-pytorch>=0.1.2",
        "timm>=0.3.2",
        "numpy>=1.19.2",
		"tensorboard>=2.5",
        "moviepy>=1.0.3"
    ],
    packages=find_packages())
