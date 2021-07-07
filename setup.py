#!/usr/bin/env python
"""
Author: Ryo Yonetani
Affiliation:  OSX
"""

from setuptools import setup, find_packages

setup(
    name="planning_experiment",
    version="0.1.0",
    description="Path Planning using Neural A* Search",
    author="Ryo Yonetani",
    author_email="ryo.yonetani@sinicx.com",
    url="https://github.com/omron-sinicx/neural-astar",
    install_requires=[
        "torch==1.5.0",
        "torchvision==0.6.0",
        "numpy==1.18.4",
        "matplotlib==3.2.1",
        "bootstrapped==0.0.2",
        "tqdm==4.42.1",
        "gin-config==0.3.0",
        "natsort==7.0.1",
        "pytorch3d==0.2.0",
        "pqdict==1.1.0",
        "ipython==7.13.0",
        "jupyterlab==2.1.2",
	    "scikit-image==0.17.2"
    ],
    packages=find_packages())
