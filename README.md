# Path Planning using Neural A\* Search (ICML 2021)

This is a repository for the following paper:

Yonetani, Taniai, Barekatain, Nishimura, Kanezaki, "Path Planning using Neural A\* Search", ICML, 2021 [[paper]](https://arxiv.org/abs/2009.07476) [[project page]](https://omron-sinicx.github.io/neural-astar/)

## TL;DR

Neural A\* is a novel data-driven search-based planner that consists of a trainable encoder and a differentiable version of A\* search algorithm called differentiable A* module. Neural A\* learns from demonstrations to improve the trade-off between search optimality and efficiency in path planning and also to enable the planning directly on raw image inputs.

| A\* search | Neural A\* search | 
|:--:|:--:|
| ![astar](assets/astar.gif) | ![neural_astar](assets/neural_astar.gif)|


## Overview
- This branch presents a minimal example for training and evaluating Neural A* on shortest path problems.
- For reproducing experiments in our ICML'21 paper, please refer to `icml2021` branch (TBA).
- For creating datasets used in our experiments, please visit `planning datasets` repository (TBA).

## Getting started
- The code has been tested on Ubuntu 18.04.5 LTS.
- Try Neural A* on Google Colab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omron-sinicx/neural-astar/blob/minimal/example.ipynb)
- See also `docker-compose.yml` and `docker/Dockerfile` to reproduce our environment.


## Citation

```
# ICML2021 version (to appear)

# arXiv version
@article{DBLP:journals/corr/abs-2009-07476,
  author    = {Ryo Yonetani and
               Tatsunori Taniai and
               Mohammadamin Barekatain and
               Mai Nishimura and
               Asako Kanezaki},
  title     = {Path Planning using Neural A* Search},
  journal   = {CoRR},
  volume    = {abs/2009.07476},
  year      = {2020},
  url       = {https://arxiv.org/abs/2009.07476},
  archivePrefix = {arXiv},
  eprint    = {2009.07476},
  timestamp = {Wed, 23 Sep 2020 15:51:46 +0200},
  biburl    = {https://dblp.org/rec/journals/corr/abs-2009-07476.bib},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```
