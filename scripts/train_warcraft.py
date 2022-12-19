"""Training Neural A* 
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
import torch
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_warcraft_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint


@hydra.main(config_path="config", config_name="train_warcraft")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_warcraft_dataloader(
        config.dataset, "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_warcraft_dataloader(
        config.dataset, "val", config.params.batch_size, shuffle=False
    )

    neural_astar = NeuralAstar(
        encoder_input=config.encoder.input,
        encoder_arch=config.encoder.arch,
        encoder_depth=config.encoder.depth,
        const=config.encoder.const,
        learn_obstacles=True,
        Tmax=config.Tmax,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor="metrics/val_loss", save_weights_only=True, mode="min"
    )

    module = PlannerModule(neural_astar, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        log_every_n_steps=1,
        default_root_dir=logdir,
        max_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
