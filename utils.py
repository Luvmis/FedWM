import argparse
import functools
import os
import pathlib
import sys
import copy
from typing import List, Dict
import pickle

os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import ruamel.yaml as yaml

sys.path.append(str(pathlib.Path(__file__).parent))

import exploration as expl
import models
import tools
import envs.wrappers as wrappers
from parallel import Parallel, Damy

import torch
from torch import nn
from torch import distributions as torchd
import random

to_np = lambda x: x.detach().cpu().numpy()

def _state_dict_to_vector(state_dict):

    parts = []
    for k, v in state_dict.items():
        if isinstance(v, torch.Tensor):
            parts.append(v.detach().cpu().float().view(-1))
        else:
            parts.append(torch.tensor(v).float().view(-1))
    if len(parts) == 0:
        return torch.tensor([], dtype=torch.float32)
    return torch.cat(parts)

def _vector_to_state_dict(template_state, vector):

    out = {}
    idx = 0
    for k, v in template_state.items():
        size = int(torch.tensor(v).numel()) if not isinstance(v, torch.Tensor) else v.numel()
        slice_v = vector[idx: idx + size]
        out[k] = slice_v.view_as(torch.tensor(v)).cpu()
        idx += size
    return out

def _mix_state_dicts(sd_a, sd_b, alpha):
    out = {}
    for k in sd_a.keys():
        a = sd_a[k]
        b = sd_b[k] 
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)
        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)
        out[k] = (alpha * a + (1.0 - alpha) * b).cpu()
    return out

def _zero_like_state(self, state_dict):
    out = {}
    for k, v in state_dict.items():
        t = torch.tensor(v) if not isinstance(v, torch.Tensor) else v
        out[k] = torch.zeros_like(t).cpu()
    return out


def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset

def make_env(config, mode, id, no_iid=True, seed=0):
    suite, task = config.task.split("_", 1)
    if suite == "dmc":
        import envs.dmc as dmc

        env = dmc.DeepMindControl(
            task, config.action_repeat, config.size, seed=config.seed + id
        )
        env = wrappers.NormalizeActions(env)
    elif suite == "atari":
        import envs.atari as atari

        mode = None
        difficulty = None
        if no_iid:
            models, difficultys = tools.get_modde_difficulty(task, seed)
            mode = random.choice(models)
            difficulty = random.choice(difficultys)

        with open(os.path.join(config.logdir, 'game.txt'), 'a', encoding='utf-8') as f:
            f.write(f'{task} difficulty: {difficulty}\n')
            f.write(f'{task} mode: {mode}\n')

        env = atari.Atari(
            task,
            config.action_repeat,
            config.size,
            gray=config.grayscale,
            noops=config.noops,
            lives=config.lives,
            sticky=config.stickey,
            actions=config.actions,
            resize=config.resize,
            seed=config.seed + id,
            mode=mode,
            difficulty=difficulty,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "dmlab":
        import envs.dmlab as dmlab

        env = dmlab.DeepMindLabyrinth(
            task,
            mode if "train" in mode else "test",
            config.action_repeat,
            seed=config.seed + id,
        )
        env = wrappers.OneHotAction(env)
    elif suite == "memorymaze":
        from envs.memorymaze import MemoryMaze

        env = MemoryMaze(task, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "crafter":
        import envs.crafter as crafter

        env = crafter.Crafter(task, config.size, seed=config.seed + id)
        env = wrappers.OneHotAction(env)
    elif suite == "minecraft":
        import envs.minecraft as minecraft

        env = minecraft.make_env(task, size=config.size, break_speed=config.break_speed)
        env = wrappers.OneHotAction(env)
    else:
        raise NotImplementedError(suite)
    env = wrappers.TimeLimit(env, config.time_limit)
    env = wrappers.SelectAction(env, key="action")
    env = wrappers.UUID(env)
    if suite == "minecraft":
        env = wrappers.RewardObs(env)
    return env


def effective_rank_safe(z, eps=1e-6, max_dim=128):
    z = z.reshape(-1, z.shape[-1])

    if z.shape[0] < 2:
        return 0.0

    if z.shape[1] > max_dim:
        z = z[:, :max_dim]

    z = z - z.mean(0, keepdim=True)

    s = torch.linalg.svdvals(z)
    s = s.clamp(min=eps)

    p = s / s.sum()
    entropy = -(p * torch.log(p)).sum()

    return torch.exp(entropy).item()


# ==========================Encoder Decoder=============================

def encoder_weight(stats, gamma=5.0, tau=0.5):
    rank = stats["rank"]
    entropy = stats["entropy"]

    # Quality
    Q = rank

    # Stability gate
    gate = torch.sigmoid(
        torch.tensor(gamma * (entropy - tau))
    ).item()

    diversity = max(0.0, 1.0 - abs(entropy - 0.5) * 2)

    return Q * gate * diversity


def decoder_weight(stats, sigma=1.0):
    recon = stats["recon"]
    return float(torch.exp(torch.tensor(-recon / sigma)))


def normalize(ws, eps=1e-8):
    s = sum(ws) + eps
    return [w / s for w in ws]


# utils.py

import torch
import numpy as np


def effective_rank_safe(feature_matrix, epsilon=1e-8):
    if feature_matrix.dim() > 2:
        feature_matrix = feature_matrix.reshape(-1, feature_matrix.shape[-1])


    Z = feature_matrix.float()
    n, d = Z.shape

    if n < 2 or d < 1:
        return 1.0

    try:
        Z_centered = Z - Z.mean(dim=0, keepdim=True)
        cov_matrix = (Z_centered.T @ Z_centered) / n

        eigenvalues = torch.linalg.eigvalsh(cov_matrix)
        eigenvalues = eigenvalues.flip(0)  #

        eigenvalues = eigenvalues[eigenvalues > epsilon]

        if len(eigenvalues) == 0:
            return 1.0

        lambda_sum = eigenvalues.sum()
        normalized_eigenvalues = eigenvalues / (lambda_sum + epsilon)

        valid_lambdas = normalized_eigenvalues[normalized_eigenvalues > epsilon]
        spectral_entropy = -(valid_lambdas * torch.log(valid_lambdas)).sum()

        erank = torch.exp(spectral_entropy)

        return erank.item()

    except Exception as e:
        print(f"Warning: ERank computation failed: {e}")
        return 1.0


def compute_feature_entropy(feature_matrix, epsilon=1e-8):
    if feature_matrix.dim() > 2:
        feature_matrix = feature_matrix.reshape(-1, feature_matrix.shape[-1])

    std_per_dim = feature_matrix.std(dim=0)

    std_normalized = std_per_dim / (std_per_dim.sum() + epsilon)

    entropy = -(std_normalized * torch.log(std_normalized + epsilon)).sum()

    return entropy.item()