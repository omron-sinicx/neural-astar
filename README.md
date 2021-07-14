# Path Planning using Neural A\* Search (ICML 2021)

This is a repository for the following paper:

Ryo Yonetani*, Tatsunori Taniai*, Mohammadamin Barekatain, Mai Nishimura, Asako Kanezaki, "Path Planning using Neural A\* Search", ICML, 2021 [[paper]](https://arxiv.org/abs/2009.07476) [[project page]](https://omron-sinicx.github.io/neural-astar/)

## TL;DR

Neural A\* is a novel data-driven search-based planner that consists of a trainable encoder and a differentiable version of A\* search algorithm called differentiable A* module. Neural A\* learns from demonstrations to improve the trade-off between search optimality and efficiency in path planning and also to enable the planning directly on raw image inputs.

| A\* search | Neural A\* search | 
|:--:|:--:|
| ![astar](assets/astar.gif) | ![neural_astar](assets/neural_astar.gif)|


## Overview
- This branch presents a minimal example for training and evaluating Neural A* on shortest path problems.
- For reproducing experiments in our ICML'21 paper, please refer to [icml2021](https://github.com/omron-sinicx/neural-astar/tree/icml2021) branch.
- For creating datasets used in our experiments, please visit [planning datasets](https://github.com/omron-sinicx/planning-datasets) repository.

## Getting started
- The code has been tested on Ubuntu 18.04.5 LTS.
- Try Neural A* on Google Colab! [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/omron-sinicx/neural-astar/blob/minimal/example.ipynb)
- See also `docker-compose.yml` and `docker/Dockerfile` to reproduce our environment.


## Citation

```
# ICML2021 version
@InProceedings{pmlr-v139-yonetani21a,
  title =      {Path Planning using Neural A* Search},
  author    = {Ryo Yonetani and
               Tatsunori Taniai and
               Mohammadamin Barekatain and
               Mai Nishimura and
               Asako Kanezaki},
  booktitle =      {Proceedings of the 38th International Conference on Machine Learning},
  pages =      {12029--12039},
  year =      {2021},
  editor =      {Meila, Marina and Zhang, Tong},
  volume =      {139},
  series =      {Proceedings of Machine Learning Research},
  month =      {18--24 Jul},
  publisher =    {PMLR},
  pdf =      {http://proceedings.mlr.press/v139/yonetani21a/yonetani21a.pdf},
  url =      {http://proceedings.mlr.press/v139/yonetani21a.html},
}

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

## Acknowledgments
This repository includes some code from [RLAgent/gated-path-planning-networks](https://github.com/RLAgent/gated-path-planning-networks) [1] with permission of the authors and from [martius-lab/blackbox-backprop](https://github.com/martius-lab/blackbox-backprop) [2].

## References
- [1] [Lisa Lee*, Emilio Parisotto*, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov, "Gated Path Planning Networks", ICML, 2018.](https://arxiv.org/abs/1806.06408)
- [2] [Marin Vlastelica Pogančić, Anselm Paulus, Vit Musil, Georg Martius, Michal Rolinek, "Differentiation of Blackbox Combinatorial Solvers", ICLR, 2020.](https://arxiv.org/abs/1912.02175)
