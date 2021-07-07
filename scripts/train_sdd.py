"""Train a planner model for SDD
Author: Ryo Yonetani
Affiliation: OMRON SINIC X
"""

import sys
import os
import numpy as np
import argparse

import gin
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from pytorch3d.loss import chamfer_distance as p3chamfer

from planning_experiment.planners import NeuralAstar, BBAstar
from planning_experiment.utils import get_mechanism, SDD, set_global_seeds


def chamfer_distance(paths, traj_images):
    cd = torch.zeros(len(paths))
    for i in range(len(paths)):
        pts1 = torch.stack(torch.where(paths[i][0])).T.float()
        pts2 = torch.stack(torch.where(traj_images[i][0])).T.float()
        cd[i] = p3chamfer(pts1.unsqueeze(0).cpu(), pts2.unsqueeze(0).cpu())[0]
    return cd


def main(args):
    # config
    set_global_seeds(args.seed)
    device = (torch.device("cuda")
              if torch.cuda.is_available() else torch.device("cpu"))
    gin_files = args.gin_files
    gin_bindings = [] if args.gin_bindings is None else args.gin_bindings
    gin.parse_config_files_and_bindings(gin_files, gin_bindings)
    save_directory = os.path.join(
        args.save_dir,
        '%s_%s_%0.2f_%04d' % (args.test_scene, args.mechanism,
                              float(args.hardness), int(args.max_steps)),
        '%s_%s_%d' %
        (args.model_name, args.encoder_arch, int(args.encoder_depth)))
    os.makedirs(save_directory, exist_ok=True)

    # model
    model_name = getattr(sys.modules[__name__], args.model_name)
    model = model_name(get_mechanism(args.mechanism),
                       encoder_input=args.encoder_input,
                       encoder_arch=args.encoder_arch,
                       encoder_depth=int(args.encoder_depth),
                       learn_obstacles=True,
                       ignore_obstacles=True).to(device)
    if args.eval_only:
        print("eval only")
        model.load_state_dict(torch.load(save_directory + "/data.pth"))
        EVAL_ONLY = True
        args.num_epochs = 1
    else:
        EVAL_ONLY = False

    # optimizer
    opt = optim.RMSprop(model.parameters(), lr=0.001)

    # dataloaders
    hardness = SDD(args.data_dir,
                   is_train=True,
                   test_scene=args.test_scene,
                   load_hardness=True)
    hardness = np.array([x for x in hardness])
    train_dataset = SDD(args.data_dir,
                        is_train=True,
                        test_scene=args.test_scene,
                        load_hardness=False)
    train_dataset = Subset(train_dataset,
                           np.where(hardness <= float(args.hardness))[0])
    train_loader = DataLoader(train_dataset,
                              batch_size=int(args.batch_size),
                              shuffle=True,
                              num_workers=0)

    test_dataset = SDD(args.data_dir,
                       is_train=False,
                       test_scene=args.test_scene,
                       load_hardness=False)
    test_loader = DataLoader(test_dataset,
                             batch_size=int(args.batch_size),
                             shuffle=False,
                             num_workers=0)

    for e in range(int(args.num_epochs)):
        for train_flag, dataloader in zip([True, False],
                                          [train_loader, test_loader]):
            if train_flag:
                model.train()
            else:
                model.eval()

            total_loss = []
            total_cd = []
            total_lr = []
            for i, samples in enumerate(dataloader):
                images = samples['image'].to(device)
                start_images = samples['start_image'].to(device)
                goal_images = samples['goal_image'].to(device)
                traj_images = samples['traj_image'].to(device)
                length_ratio = samples['length_ratio'].float()

                if model.training & (EVAL_ONLY is not True):
                    histories, paths, pred_costs = model(
                        images, start_images, goal_images)
                    opt.zero_grad()
                    loss = nn.L1Loss()(histories, traj_images)
                    loss.backward()
                    opt.step()
                    cd = chamfer_distance(paths, traj_images)
                else:
                    with torch.no_grad():
                        histories, paths, pred_costs = model(
                            images, start_images, goal_images)
                        loss = nn.L1Loss()(histories, traj_images)
                        cd = chamfer_distance(paths, traj_images)

                total_loss.append(loss.item())
                total_cd.append(cd)
                total_lr.append(length_ratio)

                sys.stdout.write("\r(%s) %d/%d:%d/%d loss %0.2e cd: %d\r" % (
                    "train" if model.training else "test",
                    e,
                    int(args.num_epochs),
                    i,
                    len(dataloader),
                    loss.data.item(),
                    cd.mean(),
                ))
                sys.stdout.flush()
            total_loss = np.array(total_loss)
            total_cd = np.concatenate(total_cd)
            total_lr = np.concatenate(total_lr)
            print("\n(%s) %d/%d loss %0.4f\n" %
                  ("train" if model.training else "test", e,
                   int(args.num_epochs), total_loss.mean()))
            if train_flag:
                train_loss = total_loss
                train_cd = total_cd
                train_lr = total_lr
            else:
                test_loss = total_loss
                test_cd = total_cd
                test_lr = total_lr

    torch.save(model.state_dict(), save_directory + "/data.pth")
    np.savez_compressed(save_directory + "/loss.npz",
                        train_loss=train_loss,
                        test_loss=test_loss,
                        train_cd=train_cd,
                        test_cd=test_cd,
                        train_lr=train_lr,
                        test_lr=test_lr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gin-files",
        "-gf",
        type=str,
        required=True,
        nargs="+",
        help="gin config file(s)",
    )
    parser.add_argument("--gin-bindings",
                        "-gb",
                        type=str,
                        nargs="+",
                        help="gin extra binding(s)")
    parser.add_argument("--model-name",
                        "-m",
                        type=str,
                        required=True,
                        choices=["NeuralAstar", "BBAstar"])
    parser.add_argument("--mechanism",
                        "-mc",
                        type=str,
                        default="moore",
                        choices=["news", "moore"])
    parser.add_argument("--encoder-arch",
                        "-ea",
                        type=str,
                        default="UnetMlt",
                        choices=["Unet", "UnetMlt"])
    parser.add_argument("--encoder-depth", "-ed", type=int, default=4)
    parser.add_argument("--encoder-input", "-ei", type=str, default='rgb+')
    parser.add_argument("--data-dir",
                        "-d",
                        type=str,
                        default="data/sdd/s064_0.5_128_150")
    parser.add_argument("--save-dir",
                        "-s",
                        type=str,
                        default="log/icml2021_sdd")
    parser.add_argument("--test-scene", "-t", type=str, default="video0")
    parser.add_argument("--hardness", "-hd", type=float, default=1.0)
    parser.add_argument("--batch-size", "-b", type=int, default=64)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--num-epochs", "-n", type=int, default=20)
    parser.add_argument("--max-steps", "-ms", type=int, default=300)
    parser.add_argument("--eval-only", action="store_true")

    args = parser.parse_args()
    main(args)
