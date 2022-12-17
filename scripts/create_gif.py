import json
import os

import hydra
import moviepy.editor as mpy
import torch
from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import visualize_results


@hydra.main(config_path="config", config_name="create_gif")
def main(config):
    dataname = os.path.basename(config.dataset)

    if config.planner == "na":
        planner = NeuralAstar(encoder_arch=config.encoder)
        planner.load_state_dict(torch.load(f"{config.modeldir}/{dataname}/best.pt"))
    else:
        planner = VanillaAstar()

    problem_id = config.problem_id
    savedir = f"{config.resultdir}/{config.planner}"
    os.makedirs(savedir, exist_ok=True)

    dataloader = create_dataloader(
        config.dataset + ".npz",
        "test",
        100,
        shuffle=False,
        num_starts=1,
    )
    map_designs, start_maps, goal_maps, opt_trajs = next(iter(dataloader))
    outputs = planner(
        map_designs, start_maps, goal_maps, store_intermediate_results=True
    )

    outputs = planner(
        map_designs[problem_id : problem_id + 1],
        start_maps[problem_id : problem_id + 1],
        goal_maps[problem_id : problem_id + 1],
        store_intermediate_results=True,
    )
    frames = [
        visualize_results(
            map_designs[problem_id : problem_id + 1], intermediate_results, scale=4
        )
        for intermediate_results in outputs.intermediate_results
    ]
    clip = mpy.ImageSequenceClip(frames + [frames[-1]] * 15, fps=30)
    clip.write_gif(f"{savedir}/video_{dataname}_{problem_id:04d}.gif")


if __name__ == "__main__":
    main()
