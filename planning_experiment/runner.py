"""Runs a planner model on a given dataset and records statistics.
Author: Mohammadamin Barekatain, Ryo Yonetani
Affiliation: OMRON SINIC X
Small parts of this script has been copied from https://github.com/RLAgent/gated-path-planning-networks
"""

import copy
import sys
import re

import gin
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .utils.mechanism import NorthEastWestSouth, Moore
from .utils.utils import load_state_dict

gin.external_configurable(optim.RMSprop, "RMSprop", blacklist=["params", "lr"])
gin.external_configurable(optim.Adam, "Adam", blacklist=["params", "lr"])
gin.external_configurable(optim.SGD, "SGD", blacklist=["params", "lr"])


@gin.configurable
class Runner:
    """
    The Runner class runs a planner model on a given dataset and records
    statistics such as loss, % Optimal, and % Success.
    """
    def __init__(
        self,
        planner=gin.REQUIRED,
        mechanism=gin.REQUIRED,
        optimizer_cls=gin.REQUIRED,
        lr=gin.REQUIRED,
        scheduler_params=None,
        clip_grad=40.0,
        best_model_metric="loss",
        pretrained_path="",
        multi_gpu=False,
    ):
        """
        Args:
        """
        self.multi_gpu = multi_gpu
        self.clip_grad = clip_grad
        self.scheduler_params = scheduler_params
        self.best_model_metric = best_model_metric
        self.mechanism = mechanism

        # Instantiate the model
        self.planner = planner(mechanism=mechanism)
        self.model_name = re.sub(
            "Planner|\@", "",
            gin.query_parameter("Runner.planner").__repr__())
        # Use GPU if available
        self.device = (torch.device("cuda")
                       if torch.cuda.is_available() else torch.device("cpu"))
        if self.multi_gpu:
            self.planner.model = nn.DataParallel(self.planner.model)
            self.planner.astar_ref = nn.DataParallel(self.planner.astar_ref)
        self.planner.model = self.planner.model.to(self.device)
        self.planner.astar_ref = self.planner.astar_ref.to(self.device)
        self._train = self.planner.train_self
        self._eval = self.planner.eval_self

        self.optimizer = optimizer_cls(self.planner.model.parameters(), lr)
        if self.scheduler_params is not None:
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optimizer, **self.scheduler_params)

        self.best_state_dict = {}
        self.last_state_dict = {}

        # Load model from file if provided
        if pretrained_path != "":
            checkpoint = torch.load(pretrained_path)
            state_dict = checkpoint["best_state_dict"]
            self.best_state_dict = state_dict
            self.last_state_dict = state_dict
            load_state_dict(self.planner.model, state_dict, self.multi_gpu)

        # Track the best performing model so far
        self.best_score = -1 * np.inf

    def _run(self, dataloader, train=False, store_best=False, save_file=False):
        """
        Runs the model on the given data.
        Args:
          dataloader (torch.utils.data.Dataset): Dataset loader
          train (bool): Whether to train the model
          store_best (bool): Whether to store the best model
          save_file (bool): Whether to save the predicted distances. If None, do not save.
        Returns:
          info (dict): Performance statistics, including
          info["avg_loss"] (float): Average loss
          info["avg_optimal"] (float): Average % Optimal
          info["avg_success"] (float): Average % Success
          info["avg_exp"] (float): Average ratio of number of expansions.
          info["weight_norm"] (float): Model weight norm, stored if train=True
          info["grad_norm"]: Gradient norm, stored if train=True
          info["is_best"] (bool): Whether the model is best, stored if store_best=True
        """
        model = self.planner.model
        info = {
            "avg_loss": 0.0,
            "avg_optimal": 0.0,
            "avg_success": 0.0,
            "avg_exp": 0.0
        }
        num_batches = 0

        if save_file:
            predictions = {
                "pred_dists": [],
                "rel_exps": [],
                "opt_dists": [],
                "masks": [],
            }

        for i, data in enumerate(dataloader):
            # Get input batch.
            map_designs, goal_maps, opt_policies, opt_dists = data

            # Have a copy in CPU
            map_designs_CPU = map_designs.data.numpy()
            goal_maps_CPU = goal_maps.data.numpy()
            opt_policies_CPU = opt_policies.data.numpy()
            opt_dists_CPU = opt_dists.data.numpy()

            map_designs = map_designs.to(self.device)
            goal_maps = goal_maps.to(self.device)
            opt_policies = opt_policies.to(self.device)
            opt_dists = opt_dists.to(self.device)

            # Reshape batch-wise if necessary
            if map_designs.dim() == 3:
                map_designs = map_designs.unsqueeze(1)
                map_designs_CPU = np.expand_dims(map_designs_CPU, axis=1)

            if train:
                loss, p_opt, p_suc, p_exp = self._train(
                    map_designs,
                    goal_maps,
                    opt_policies,
                    opt_dists,
                    map_designs_CPU,
                    goal_maps_CPU,
                    opt_policies_CPU,
                    opt_dists_CPU,
                    self.device,
                )

                # Skip training if the planner is VanillaAstar
                if self.model_name == "VanillaAstar":
                    info["lr"] = -1
                else:
                    self.optimizer.zero_grad()
                    loss.backward()

                    # Clip the gradient norm
                    if self.clip_grad:
                        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                                       self.clip_grad)

                    # Update parameters
                    self.optimizer.step()
                    if self.scheduler_params is not None:
                        self.scheduler.step()
                    info["lr"] = self.optimizer.param_groups[0]["lr"]

            else:
                with torch.no_grad():
                    num_eval_points = 5 if save_file else 2  # save time for validation
                    # Compute success and optimality
                    loss, p_opt, p_suc, p_exp, pred_dists, rel_exps, masks = self._eval(
                        map_designs,
                        goal_maps,
                        opt_policies,
                        opt_dists,
                        map_designs_CPU,
                        goal_maps_CPU,
                        opt_policies_CPU,
                        opt_dists_CPU,
                        self.device,
                        num_eval_points,
                    )
                if save_file:
                    predictions["pred_dists"].append(pred_dists)
                    predictions["rel_exps"].append(rel_exps)
                    predictions["opt_dists"].append(opt_dists_CPU)
                    predictions["masks"].append(masks)

            info["avg_loss"] += loss.data.item()
            info["avg_optimal"] += p_opt
            info["avg_success"] += p_suc
            info["avg_exp"] += p_exp
            num_batches += 1
            # print log
            sys.stdout.write(
                "\r(%s) %d/%d loss %0.2e, p_opt %0.2f p_suc %0.2f p_exp %0.2f\r"
                % (
                    "train" if model.training else "test",
                    i,
                    len(dataloader),
                    loss.data.item(),
                    p_opt,
                    p_suc,
                    p_exp,
                ))
            sys.stdout.flush()

        info["avg_loss"] = info["avg_loss"] / num_batches
        info["avg_optimal"] = info["avg_optimal"] / num_batches
        info["avg_success"] = info["avg_success"] / num_batches
        info["avg_exp"] = info["avg_exp"] / num_batches

        info["weight_norm"] = 0.0
        info["grad_norm"] = 0.0
        if train & (self.model_name != "VanillaAstar"):
            # Calculate weight norm
            weight_norm, grad_norm = 0, 0
            for p in model.parameters():
                weight_norm += torch.norm(p)**2
                grad_norm += torch.norm(p.grad)**2 if p.grad is not None else 0
            info["weight_norm"] = float(
                np.sqrt(weight_norm.cpu().data.numpy().item()))
            info["grad_norm"] = float(
                np.sqrt(grad_norm.cpu().data.numpy().item()))

            self.last_state_dict = (model.module.state_dict()
                                    if self.multi_gpu else model.state_dict())

        elif store_best:
            # Was the validation accuracy greater than the best one?
            if self.best_model_metric == "suc":
                score = info["avg_success"]
            elif self.best_model_metric == "opt":
                score = info["avg_optimal"]
            elif self.best_model_metric == "exp":
                score = -1 * info["avg_exp"]
            elif self.best_model_metric == "hmean":
                score = 2. / (1. / info["avg_optimal"] + 1. /
                              (1 - np.minimum(0.9999, info["avg_exp"])))
            else:
                score = -1 * info["avg_loss"]
            if score >= self.best_score:
                self.best_score = score
                self.best_state_dict = copy.deepcopy(model.module.state_dict(
                ) if self.multi_gpu else model.state_dict())
                info["is_best"] = True
            else:
                info["is_best"] = False

        if save_file:
            for k in predictions.keys():
                predictions[k] = np.concatenate(predictions[k])
            info["predictions"] = predictions

        return info

    def train(self, dataloader):
        """
        Trains the model on the given training dataset.
        """
        self.planner.model.train()
        return self._run(dataloader, train=True)

    def validate(self, dataloader):
        """
        Evaluates the model on the given validation dataset. Stores the
        current model if it achieves the best validation performance.
        """
        self.planner.model.eval()
        return self._run(dataloader, store_best=True)

    def test(self, dataloader, load_best=False):
        """
        Tests the model on the given dataset.
        """

        if "VanillaAstar" not in self.planner.__repr__():
            state_dict = self.best_state_dict if load_best else self.last_state_dict
            load_state_dict(self.planner.model, state_dict, self.multi_gpu)

        self.planner.model.eval()
        return self._run(dataloader, store_best=False, save_file=True)
