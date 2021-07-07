# Path Planning using Neural A\* Search (ICML 2021)

This is a repository for the following paper:

Yonetani, Taniai, Barekatain, Nishimura, Kanezaki, "Path Planning using Neural A\* Search", ICML, 2021 [[paper]](https://arxiv.org/abs/2009.07476) [[project page]](https://omron-sinicx.github.io/neural-astar/)

## TL;DR

Neural A\* is a novel data-driven search-based planner that consists of a trainable encoder and a differentiable version of A\* search algorithm called differentiable A* module. Neural A\* learns from demonstrations to improve the trade-off between search optimality and efficiency in path planning and also to enable the planning directly on raw image inputs.

## Overview
- This branch provides the code to reproduce the experiments in our ICML paper.
- For a minimal example to train and evaluate Neural A* on shortest path problems, please refer to [minimal](https://github.com/omron-sinicx/neural-astar/tree/minimal) branch.
- For creating datasets used in our experiments, please visit [planning datasets](https://github.com/omron-sinicx/planning-datasets) repository.

## Getting started
- The code has been tested on Ubuntu 18.04.3 LTS.
- Use `docker-compose.yml` and `docker/Dockerfile` to reproduce our environment.

### Training Neural A* and baseline models on MP/TiledMP/CSM datasets

Download all the data from [planning datasets](https://github.com/omron-sinicx/planning-datasets) and place them in `scripts/data`

```sh
data
├── mpd
│   ├── all_064_moore_c16.npz
│   ├── alternating_gaps_032_moore_c8.npz
│   ├── bugtrap_forest_032_moore_c8.npz
│   ├── forest_032_moore_c8.npz
│   ├── gaps_and_forest_032_moore_c8.npz
│   ├── mazes_032_moore_c8.npz
│   ├── multiple_bugtraps_032_moore_c8.npz
│   ├── original
│   ├── shifting_gaps_032_moore_c8.npz
│   └── single_bugtrap_032_moore_c8.npz
├── sdd
│   ├── original
│   └── s064_0.5_128_300
└── street
    ├── mixed_064_moore_c16.npz
    └── original
```

Run all scripts from the `scripts` directory.
```sh
$ cd scripts
$ sh 0_MP.sh | sh
$ sh 1_TiledMP.sh | sh
$ sh 2_CSM.sh | sh
```

If you want to parallelize experiments, use [GNU Parallel](https://www.gnu.org/software/parallel/): 

```sh
$ sh 0_MP.sh | parallel -j 2 --ungroup
```

Once all the training sessions are done, check results with the following commands:

```sh
# show the opt, exp, and hmean scores
$ python show_results.py
# visualize some results and save them in `figures`
$ python visualize_results.py
```

### Training Neural A* and baseline models on Stanford Drone Dataset

```sh
$ sh 3_SDD.sh | sh
```

Once all the training sessions are done, check results with the following commands:

```sh
# show the chamfer distance scores
$ python show_results_sdd.py
# visualize some results and save them in `figures`
$ python visualize_results_sdd.py
```

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

## Acknowledgments
This repository includes some code from [RLAgent/gated-path-planning-networks](https://github.com/RLAgent/gated-path-planning-networks) [1] with permission of the authors and from [martius-lab/blackbox-backprop](https://github.com/martius-lab/blackbox-backprop) [2].

## References
- [1] [Lisa Lee*, Emilio Parisotto*, Devendra Singh Chaplot, Eric Xing, Ruslan Salakhutdinov, "Gated Path Planning Networks", ICML, 2018.](https://arxiv.org/abs/1806.06408)
- [2] [Marin Vlastelica Pogančić, Anselm Paulus, Vit Musil, Georg Martius, Michal Rolinek, "Differentiation of Blackbox Combinatorial Solvers", ICLR, 2020.](https://arxiv.org/abs/1912.02175)