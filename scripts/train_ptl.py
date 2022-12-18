"""Training Neural A* 
Author: Ryo Yonetani
Affiliation: OSX
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as pl
from neural_astar.planner import NeuralAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import PlannerModule, set_global_seeds
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.tensorboard import SummaryWriter


@hydra.main(config_path="config", config_name="train")
def main(config):

    # dataloaders
    set_global_seeds(config.seed)
    train_loader = create_dataloader(
        config.dataset + ".npz", "train", config.params.batch_size, shuffle=True
    )
    val_loader = create_dataloader(
        config.dataset + ".npz", "valid", config.params.batch_size, shuffle=False
    )
    test_loader = create_dataloader(
        config.dataset + ".npz", "test", config.params.batch_size, shuffle=False
    )

    neural_astar = NeuralAstar(encoder_arch=config.encoder, Tmax=config.Tmax)
    checkpoint_callback = ModelCheckpoint(monitor="val_h_mean")

    module = PlannerModule(neural_astar, config)
    logdir = f"{config.logdir}/{os.path.basename(config.dataset)}"
    trainer = pl.Trainer(
        default_root_dir=logdir,
        min_epochs=config.params.num_epochs,
        callbacks=[checkpoint_callback],
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
