"""Training Neural A* 
Author: Ryo Yonetani
Affiliation: OSX
"""

import argparse
from datetime import datetime
import subprocess

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from neural_astar.planner import NeuralAstar, VanillaAstar
from neural_astar.utils.data import create_dataloader
from neural_astar.utils.training import (set_global_seeds, run_planner,
                                         calc_metrics, visualize_results,
                                         Metrics)


def main():

    # argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_npz",
                        "-d",
                        type=str,
                        default="data/mazes_032_moore_c8.npz")
    parser.add_argument("--logdir", "-l", type=str, default="log")
    parser.add_argument("--encoder-arch",
                        "-e",
                        type=str,
                        default="CNN",
                        choices=["CNN", "Unet"])
    parser.add_argument("--batch-size", "-b", type=int, default=100)
    parser.add_argument("--num-epochs", "-n", type=int, default=50)
    parser.add_argument("--seed", "-s", type=int, default=1234)
    args = parser.parse_args()

    # dataloaders
    set_global_seeds(args.seed)
    train_loader = create_dataloader(args.data_npz,
                                     "train",
                                     args.batch_size,
                                     shuffle=True)
    val_loader = create_dataloader(args.data_npz,
                                   "valid",
                                   args.batch_size,
                                   shuffle=False)
    test_loader = create_dataloader(args.data_npz,
                                    "test",
                                    args.batch_size,
                                    shuffle=False)

    # planners
    device = "cuda" if torch.cuda.is_available() else "cpu"
    neural_astar = NeuralAstar(encoder_arch=args.encoder_arch)
    neural_astar.to(device)
    vanilla_astar = VanillaAstar()
    vanilla_astar.to(device)

    # training setup
    opt = optim.RMSprop(neural_astar.parameters(), lr=0.001)
    criterion = nn.L1Loss()

    # logger setup
    now = str(datetime.now())
    logdir = f"{args.logdir}/{now}"
    writer = SummaryWriter(f"{logdir}/tb")
    h_mean_best = -1.0

    for e in range(args.num_epochs):
        train_loss, val_loss, p_opt, p_exp, h_mean = 0., 0., 0., 0., 0.

        # training
        for batch in tqdm(train_loader, desc="training", ncols=60):
            neural_astar.train()
            loss, na_outputs = run_planner(batch, neural_astar, criterion)
            train_loss += loss.item()
            opt.zero_grad()
            loss.backward()
            opt.step()
        train_loss /= len(train_loader)

        # validation
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="validation", ncols=60):
                neural_astar.eval()
                loss, na_outputs = run_planner(batch, neural_astar, criterion)
                _, va_outputs = run_planner(batch, vanilla_astar, criterion)
                metrics = calc_metrics(na_outputs, va_outputs)
                val_loss += loss
                p_opt += metrics.p_opt
                p_exp += metrics.p_exp
                h_mean += metrics.h_mean
        val_loss /= len(val_loader)
        p_opt /= len(val_loader)
        p_exp /= len(val_loader)
        h_mean /= len(val_loader)

        # logging
        print(
            f"[epoch:{e:03d}] train_loss:{train_loss:.2e}, val_loss:{val_loss:.2e}, ",
            Metrics(p_opt, p_exp, h_mean))

        writer.add_scalar("metrics/train_loss", train_loss, e)
        writer.add_scalar("metrics/val_loss", val_loss, e)
        writer.add_scalar("metrics/p_opt", p_opt, e)
        writer.add_scalar("metrics/p_exp", p_exp, e)
        writer.add_scalar("metrics/h_mean", h_mean, e)

        va_results = visualize_results(batch[0], va_outputs)
        na_results = visualize_results(batch[0], na_outputs)
        writer.add_image("vis/astar", va_results, e, dataformats="HWC")
        writer.add_image("vis/neural-astar", na_results, e, dataformats="HWC")

        # checkpointing
        if h_mean > h_mean_best:
            print(f"best score updated: {h_mean_best:0.3f} -> {h_mean:0.3f}")
            h_mean_best = h_mean
            subprocess.run(["rm", "-rf", f"{logdir}/best.pt"])
            torch.save(neural_astar.state_dict(), f"{logdir}/best.pt")
    writer.close()

    # testing
    neural_astar.load_state_dict(torch.load(f"{logdir}/best.pt"))
    p_opt, p_exp, h_mean = 0., 0., 0.
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="test", ncols=60):
            neural_astar.eval()
            loss, na_outputs = run_planner(batch, neural_astar, criterion)
            _, va_outputs = run_planner(batch, vanilla_astar, criterion)
            metrics = calc_metrics(na_outputs, va_outputs)
            p_opt += metrics.p_opt
            p_exp += metrics.p_exp
            h_mean += metrics.h_mean
        p_opt /= len(test_loader)
        p_exp /= len(test_loader)
        h_mean /= len(test_loader)

    print(f"[final performance]", Metrics(p_opt, p_exp, h_mean))
    np.savetxt(f"{logdir}/score.txt", [p_opt, p_exp, h_mean])


if __name__ == "__main__":
    main()
