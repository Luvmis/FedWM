
import argparse
import functools
import os
import pathlib
import sys
import copy
import random
import time
from typing import List, Dict
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed

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

to_np = lambda x: x.detach().cpu().numpy()

import utils


class FederatedDreamer(nn.Module):

    def __init__(self, obs_space, act_space, config, logger, dataset, client_id=0):
        super(FederatedDreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._client_id = client_id
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_train = tools.Every(batch_steps / config.train_ratio)
        self._should_pretrain = tools.Once()
        self._should_reset = tools.Every(config.reset_every)
        self._should_expl = tools.Until(int(config.expl_until / config.action_repeat))
        self._metrics = {}

        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset

        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        self._task_behavior = models.ImagBehavior(config, self._wm)

        if config.compile and os.name != "nt":
            self._wm = torch.compile(self._wm)
            self._task_behavior = torch.compile(self._task_behavior)

        reward = lambda f, s, a: self._wm.heads["reward"](f).mean()
        self._expl_behavior = dict(
            greedy=lambda: self._task_behavior,
            random=lambda: expl.Random(config, act_space),
            plan2explore=lambda: expl.Plan2Explore(config, self._wm, reward),
        )[config.expl_behavior]().to(self._config.device)

        self._local_steps = 0
        self._global_round = 0

        self._dynamics_stats_ema = {
            "pred_consistency": 0.0,
            "latent_health": 0.0,
            "env_complexity": 0.0,
        }
        self._dynamics_stats_count = 0

        self._encoder_stats_ema = {
            "erank": 0.0,
            "entropy": 0.0,
        }

        self._task_losses = {
            'reward': [],
            'image': [],
            'cont': [],
        }

        # ==================== SAM-FMGT ====================
        self._use_sam_fmgt = True if self._config.fed_aggregation == 'samfmgt' else False
        self._sam_rho = getattr(config, 'sam_rho', 0.05)
        self._sam_alpha = getattr(config, 'sam_alpha', 0.5)
        self._sam_adaptive = getattr(config, 'sam_adaptive', True)
        self._global_trajectory = []
        self._trajectory_max_len = getattr(config, 'trajectory_length', 3)
        self._current_global_params = None

        # ==================== FedProx ====================
        self._fedprox_mu = getattr(config, 'fedprox_mu', 0.01)
        self._global_params = None

        # ==================== SCAFFOLD ====================
        self._scaffold_lr = getattr(config, 'scaffold_lr', 1e-3)
        self._client_control = None
        self._server_control = None
        self._params_before_train = None
        self._local_train_steps = 0

        # ==================== FedSVRPG-M ====================
        self._reference_policy = None
        self._reference_gradient = None
        self._gradient_update_count = 0

        # ==================== FedHAPG-M ====================
        self._hapg_lambda = getattr(config, 'hapg_lambda', 0.5)
        self._global_policy_params = None

        # ==================== FedPAW-ICP ====================
        self._use_pawid = True if self._config.fed_aggregation == 'fedpawid' else False
        self._env_complexity_buffer = []
        self._repr_diversity_buffer = []
        self._latent_features_buffer = []
        self._pawid_epsilon_calib = 1e-5

        # ==================== FedSWA / FedMoSWA ====================
        self._use_fedswa = True if self._config.fed_aggregation == 'fedswa' else False
        self._use_fedmoswa = True if self._config.fed_aggregation == 'fedmoswa' else False
        self._initial_lr = getattr(config, 'local_lr', 1e-4)
        self._lr_decay_ratio = getattr(config, 'lr_decay_ratio', 0.1)
        self._local_iterations = 0
        self._total_local_iterations = getattr(config, 'local_steps', 500)
        self._current_lr = self._initial_lr

        self._print_method_config()

    def _print_method_config(self):
        method = self._config.fed_aggregation
        print(f"\n{'=' * 60}")
        print(f"Client {self._client_id}: Federated Method = {method.upper()}")
        print(f"{'=' * 60}")

        if method == 'fedprox':
            print(f"  FedProx μ: {self._fedprox_mu}")
        elif method == 'scaffold':
            print(f"  SCAFFOLD lr: {self._scaffold_lr}")
        elif method == 'fedsvrpg':
            print(f"  FedSVRPG-M: Variance Reduction Enabled")
        elif method == 'fedhapg':
            print(f"  FedHAPG-M λ: {self._hapg_lambda}")
        elif method == 'fedpawid':
            print(f"  FedPAW-ICP: Path-Aligned Information-Calibrated")
            print(f"  - Log-Robust Normalization")
        elif method == 'fedswa':
            print(f"  FedSWA: Stochastic Weight Averaging")
            print(f"  - Initial LR: {self._initial_lr}")
            print(f"  - LR Decay Ratio: {self._lr_decay_ratio}")
        elif method == 'fedmoswa':
            print(f"  FedMoSWA: Momentum + SWA")
            print(f"  - Initial LR: {self._initial_lr}")
        elif method == 'fedwm':
            print(f"  FedWM: Module-wise Aggregation")
        elif method == 'samfmgt':
            print(f"  SAM-FMGT: Sharpness-Aware + Global Trajectory")
            print(f"  - ρ (perturbation): {self._sam_rho}")
            print(f"  - α (trajectory weight): {self._sam_alpha}")
            print(f"  - Adaptive SAM: {self._sam_adaptive}")
            print(f"  - Trajectory length: {self._trajectory_max_len}")
        print(f"{'=' * 60}\n")

    def __call__(self, obs, reset, state=None, training=True):
        step = self._step
        if training:
            steps = (
                self._config.pretrain
                if self._should_pretrain()
                else self._should_train(step)
            )
            for _ in range(steps):
                self._train(next(self._dataset))
                self._update_count += 1
                self._local_steps += 1
                self._metrics["update_count"] = self._update_count

            if self._should_log(step):
                for name, values in self._metrics.items():
                    self._logger.scalar(f"client_{self._client_id}/{name}", float(np.mean(values)))
                    self._metrics[name] = []
                if self._config.video_pred_log:
                    openl = self._wm.video_pred(next(self._dataset))
                    self._logger.video(f"client_{self._client_id}/train_openl", to_np(openl))
                self._logger.write(fps=True)

        policy_output, state = self._policy(obs, state, training)

        if training:
            self._step += len(reset)
            self._logger.step = self._config.action_repeat * self._step
        return policy_output, state

    def _policy(self, obs, state, training):
        if state is None:
            latent = action = None
        else:
            latent, action = state
        obs = self._wm.preprocess(obs)
        embed = self._wm.encoder(obs)
        latent, _ = self._wm.dynamics.obs_step(latent, action, embed, obs["is_first"])
        if self._config.eval_state_mean:
            latent["stoch"] = latent["mean"]
        feat = self._wm.dynamics.get_feat(latent)
        if not training:
            actor = self._task_behavior.actor(feat)
            action = actor.mode()
        elif self._should_expl(self._step):
            actor = self._expl_behavior.actor(feat)
            action = actor.sample()
        else:
            actor = self._task_behavior.actor(feat)
            action = actor.sample()
        logprob = actor.log_prob(action)
        latent = {k: v.detach() for k, v in latent.items()}
        action = action.detach()
        if self._config.actor["dist"] == "onehot_gumble":
            action = torch.one_hot(
                torch.argmax(action, dim=-1), self._config.num_actions
            )
        policy_output = {"action": action, "logprob": logprob}
        state = (latent, action)
        return policy_output, state

    def _train(self, data):
        metrics = {}
        # SCAFFOLD / FedSWA / FedMoSWA
        if ((self._config.fed_aggregation == 'scaffold' and self._local_train_steps == 0) or
                ((self._use_fedswa or self._use_fedmoswa) and self._local_iterations == 0)):
            self._params_before_train = self._get_model_params()

        # FedSWA / FedMoSWA
        if self._use_fedswa or self._use_fedmoswa:
            self._update_cyclical_lr()
            self._apply_lr_to_optimizers()

        # ========== SAM-FMGT ==========
        if self._use_sam_fmgt:

            original_params = self._get_model_params()
            post, context, mets = self._wm._train(data)
            metrics.update(mets)
            start = post
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()
            metrics.update(self._task_behavior._train(start, reward)[-1])
            self._apply_sam_perturbation(original_params)

            post, context, mets = self._wm._train(data)
            metrics.update({f"sam_{k}": v for k, v in mets.items()})
            start = post
            metrics.update({f"sam_{k}": v for k, v in
                            self._task_behavior._train(start, reward)[-1].items()})

            self._restore_params(original_params)


            trajectory_loss = self._apply_trajectory_regularization()
            metrics['sam_fmgt_trajectory_loss'] = trajectory_loss

        else:
            post, context, mets = self._wm._train(data)
            metrics.update(mets)
            start = post
            reward = lambda f, s, a: self._wm.heads["reward"](
                self._wm.dynamics.get_feat(s)
            ).mode()

        # FedHAPG
        if self._config.fed_aggregation == 'fedhapg':
            on_policy_metrics = self._task_behavior._train(start, reward)[-1]
            off_policy_metrics = on_policy_metrics  # 简化版
            for key in on_policy_metrics:
                if isinstance(on_policy_metrics[key], (int, float)):
                    mixed_val = (self._hapg_lambda * on_policy_metrics[key] +
                                 (1 - self._hapg_lambda) * off_policy_metrics[key])
                    metrics[key] = mixed_val
                else:
                    metrics[key] = on_policy_metrics[key]
        else:
            metrics.update(self._task_behavior._train(start, reward)[-1])

        if self._config.expl_behavior != "greedy":
            mets = self._expl_behavior.train(start, context, data)[-1]
            metrics.update({"expl_" + key: value for key, value in mets.items()})


        # FedProx:
        if self._config.fed_aggregation == 'fedprox' and self._global_params is not None:
            proximal_loss = 0.0
            with torch.no_grad():
                for name, param in self._wm.named_parameters():
                    if param.requires_grad and name in self._global_params['wm']:
                        global_param = self._global_params['wm'][name]
                        proximal_loss += torch.sum((param - global_param) ** 2).item()
                        param.data.sub_(self._fedprox_mu * (param.data - global_param))

                for name, param in self._task_behavior.named_parameters():
                    if param.requires_grad and name in self._global_params['task_behavior']:
                        global_param = self._global_params['task_behavior'][name]
                        proximal_loss += torch.sum((param - global_param) ** 2).item()
                        param.data.sub_(self._fedprox_mu * (param.data - global_param))

            metrics['fedprox_proximal_loss'] = proximal_loss * self._fedprox_mu / 2.0


        if self._config.fed_aggregation == 'scaffold' and self._client_control is not None:
            scaffold_penalty = self._compute_scaffold_penalty()
            metrics['scaffold_penalty_loss'] = float(
                scaffold_penalty.item() if isinstance(scaffold_penalty, torch.Tensor) else scaffold_penalty)

        # FedSVRPG
        if self._config.fed_aggregation == 'fedsvrpg' and self._reference_gradient is not None:
            current_grads = self._get_current_gradients()
            vr_correction = self._compute_vr_correction(current_grads)
            self._apply_vr_correction(vr_correction)
            vr_norm = self._compute_gradient_norm(vr_correction)
            metrics['svrpg_vr_norm'] = vr_norm

        # FedMoSWA
        if self._use_fedmoswa and self._client_control is not None:
            self._apply_control_correction()


        if self._use_pawid:
            env_complexity = self._compute_env_complexity(post, context)
            self._env_complexity_buffer.append(env_complexity)
            if len(self._env_complexity_buffer) > 100:
                self._env_complexity_buffer = self._env_complexity_buffer[-100:]
            metrics['pawid_env_complexity'] = env_complexity

            if 'stoch' in post or 'deter' in post:
                latent = post.get('deter', post.get('stoch'))
                if isinstance(latent, torch.Tensor):
                    self._latent_features_buffer.append(latent.detach())
                    if len(self._latent_features_buffer) > 50:
                        self._latent_features_buffer = self._latent_features_buffer[-50:]

            if len(self._latent_features_buffer) >= 10 and self._update_count % 10 == 0:
                repr_diversity = self._compute_repr_diversity()
                self._repr_diversity_buffer.append(repr_diversity)
                if len(self._repr_diversity_buffer) > 10:
                    self._repr_diversity_buffer = self._repr_diversity_buffer[-10:]
                metrics['pawid_repr_diversity'] = repr_diversity


        self._local_train_steps += 1
        self._local_iterations += 1

        if self._use_fedswa or self._use_fedmoswa:
            metrics['current_lr'] = self._current_lr


        for name, value in metrics.items():
            if not name in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

        stats = self._collect_dynamics_stats(post, context)
        alpha = 0.01
        for k in self._dynamics_stats_ema:
            self._dynamics_stats_ema[k] = (
                    (1 - alpha) * self._dynamics_stats_ema[k] + alpha * stats[k]
            )
        self._dynamics_stats_count += 1

        for task_name in ['reward', 'image', 'cont']:
            metric_key = f"{task_name}_loss"
            if metric_key in metrics:
                loss_value = metrics[metric_key]
                if isinstance(loss_value, (torch.Tensor, np.ndarray)):
                    loss_value = float(loss_value.mean().item() if hasattr(loss_value, 'mean') else loss_value.item())
                else:
                    loss_value = float(loss_value)
                self._task_losses[task_name].append(loss_value)
                if len(self._task_losses[task_name]) > 100:
                    self._task_losses[task_name] = self._task_losses[task_name][-100:]

    def _apply_sam_perturbation(self, original_params: Dict):
        """
        应用 SAM 对抗扰动
        ε_w = ρ * ∇L(w) / ||∇L(w)||
        """
        with torch.no_grad():
            grad_norm = 0.0

            for name, param in self._wm.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += torch.sum(param.grad ** 2).item()

            for name, param in self._task_behavior.named_parameters():
                if param.requires_grad and param.grad is not None:
                    grad_norm += torch.sum(param.grad ** 2).item()

            grad_norm = np.sqrt(grad_norm) + 1e-12

            if self._sam_adaptive:
                param_norm = 0.0
                for name, param in self._wm.named_parameters():
                    if param.requires_grad:
                        param_norm += torch.sum(param.data ** 2).item()
                for name, param in self._task_behavior.named_parameters():
                    if param.requires_grad:
                        param_norm += torch.sum(param.data ** 2).item()
                param_norm = np.sqrt(param_norm) + 1e-12
                scale = self._sam_rho * param_norm / grad_norm
            else:
                scale = self._sam_rho / grad_norm

            for name, param in self._wm.named_parameters():
                if param.requires_grad and param.grad is not None:
                    epsilon = scale * param.grad
                    param.data.add_(epsilon)

            for name, param in self._task_behavior.named_parameters():
                if param.requires_grad and param.grad is not None:
                    epsilon = scale * param.grad
                    param.data.add_(epsilon)

            self._wm.zero_grad()
            self._task_behavior.zero_grad()

    def _restore_params(self, original_params: Dict):
        """恢复原始参数（梯度保留）"""
        with torch.no_grad():
            wm_state = self._wm.state_dict()
            for key in wm_state.keys():
                normalized_key = self._normalize_state_dict_key(key)
                if normalized_key in original_params['wm']:
                    wm_state[key].copy_(original_params['wm'][normalized_key])

            task_state = self._task_behavior.state_dict()
            for key in task_state.keys():
                normalized_key = self._normalize_state_dict_key(key)
                if normalized_key in original_params['task_behavior']:
                    task_state[key].copy_(original_params['task_behavior'][normalized_key])

    def _apply_trajectory_regularization(self) -> float:

        if self._current_global_params is None or len(self._global_trajectory) == 0:
            return 0.0

        try:
            trajectory_loss = 0.0
            total_weight = 0.0

            for idx, global_params in enumerate(self._global_trajectory):
                # 指数衰减权重: β^k，β = 0.9
                beta = 0.9
                weight = beta ** (len(self._global_trajectory) - 1 - idx)
                total_weight += weight

                distance = 0.0

                # World Model
                for name, param in self._wm.named_parameters():
                    if param.requires_grad:
                        normalized_name = self._normalize_state_dict_key(name)
                        if normalized_name in global_params['wm']:
                            global_param = global_params['wm'][normalized_name]
                            global_param = global_param.to(device=param.device, dtype=param.dtype)
                            distance += torch.sum((param - global_param) ** 2).item()

                # Task Behavior
                for name, param in self._task_behavior.named_parameters():
                    if param.requires_grad:
                        normalized_name = self._normalize_state_dict_key(name)
                        if normalized_name in global_params['task_behavior']:
                            global_param = global_params['task_behavior'][normalized_name]
                            global_param = global_param.to(device=param.device, dtype=param.dtype)
                            distance += torch.sum((param - global_param) ** 2).item()

                trajectory_loss += weight * distance

            if total_weight > 0:
                trajectory_loss /= total_weight

            if trajectory_loss > 0:
                with torch.no_grad():
                    latest_global = self._global_trajectory[-1]

                    for name, param in self._wm.named_parameters():
                        if param.requires_grad:
                            normalized_name = self._normalize_state_dict_key(name)
                            if normalized_name in latest_global['wm']:
                                global_param = latest_global['wm'][normalized_name]
                                global_param = global_param.to(device=param.device, dtype=param.dtype)
                                # 朝全局模型轻微调整
                                param.data.sub_(self._sam_alpha * 0.01 * (param.data - global_param))

                    for name, param in self._task_behavior.named_parameters():
                        if param.requires_grad:
                            normalized_name = self._normalize_state_dict_key(name)
                            if normalized_name in latest_global['task_behavior']:
                                global_param = latest_global['task_behavior'][normalized_name]
                                global_param = global_param.to(device=param.device, dtype=param.dtype)
                                param.data.sub_(self._sam_alpha * 0.01 * (param.data - global_param))

            return float(trajectory_loss)

        except Exception as e:
            print(f"⚠️  Trajectory regularization error: {e}")
            return 0.0

    def _compute_scaffold_penalty(self) -> torch.Tensor:

        if self._client_control is None or self._server_control is None:
            return torch.tensor(0.0, device=self._config.device, requires_grad=False)

        try:
            penalty_terms = []

            # ===== World Model =====
            for name, param in self._wm.named_parameters():
                if not param.requires_grad:
                    continue

                norm_name = self._normalize_state_dict_key(name)

                if norm_name not in self._client_control['wm']:
                    continue

                c_i = self._client_control['wm'][norm_name]
                c = self._server_control['wm'][norm_name]

                c_i = c_i.to(device=param.device, dtype=param.dtype).detach()
                c = c.to(device=param.device, dtype=param.dtype).detach()

                control_diff = c - c_i
                term = torch.sum(control_diff * param)
                penalty_terms.append(term)

            # ===== Task Behavior =====
            for name, param in self._task_behavior.named_parameters():
                if not param.requires_grad:
                    continue

                norm_name = self._normalize_state_dict_key(name)

                if norm_name not in self._client_control['task_behavior']:
                    continue

                c_i = self._client_control['task_behavior'][norm_name]
                c = self._server_control['task_behavior'][norm_name]

                c_i = c_i.to(device=param.device, dtype=param.dtype).detach()
                c = c.to(device=param.device, dtype=param.dtype).detach()

                control_diff = c - c_i
                term = torch.sum(control_diff * param)
                penalty_terms.append(term)

            if len(penalty_terms) == 0:
                return torch.tensor(0.0, device=self._config.device, requires_grad=False)

            total_penalty = sum(penalty_terms)

            scaled_penalty = self._scaffold_lr * total_penalty / (len(penalty_terms) + 1e-8)

            return scaled_penalty

        except Exception as e:
            print(f"❌ Error computing SCAFFOLD penalty: {e}")
            import traceback
            traceback.print_exc()
            return torch.tensor(0.0, device=self._config.device, requires_grad=False)

    # ==================== FedSWA / FedMoSWA 方法 ====================
    def _update_cyclical_lr(self):
        k = self._local_iterations
        K = self._total_local_iterations
        eta_l = self._initial_lr
        rho = self._lr_decay_ratio
        self._current_lr = eta_l * (1 - k / K) + (k / K) * rho * eta_l

    def _apply_lr_to_optimizers(self):
        if hasattr(self._wm, '_opt'):
            for param_group in self._wm._opt.param_groups:
                param_group['lr'] = self._current_lr
        if hasattr(self._task_behavior, '_opt'):
            for param_group in self._task_behavior._opt.param_groups:
                param_group['lr'] = self._current_lr

    def _apply_control_correction(self):
        with torch.no_grad():
            for name, param in self._wm.named_parameters():
                if param.requires_grad:
                    normalized_name = self._normalize_state_dict_key(name)
                    if (normalized_name in self._client_control['wm'] and
                            normalized_name in self._server_control['wm']):
                        c_i = self._client_control['wm'][normalized_name]
                        m = self._server_control['wm'][normalized_name]
                        correction = self._current_lr * (c_i - m)
                        param.data.sub_(correction)

            for name, param in self._task_behavior.named_parameters():
                if param.requires_grad:
                    normalized_name = self._normalize_state_dict_key(name)
                    if (normalized_name in self._client_control['task_behavior'] and
                            normalized_name in self._server_control['task_behavior']):
                        c_i = self._client_control['task_behavior'][normalized_name]
                        m = self._server_control['task_behavior'][normalized_name]
                        correction = self._current_lr * (c_i - m)
                        param.data.sub_(correction)

    # ==================== SCAFFOLD 方法 ====================
    def _apply_scaffold_correction(self):
        with torch.no_grad():
            wm_state = self._wm.state_dict()
            task_state = self._task_behavior.state_dict()

            for key in wm_state.keys():
                normalized_key = self._normalize_state_dict_key(key)
                if normalized_key in self._client_control['wm'] and normalized_key in self._server_control['wm']:
                    c_i = self._client_control['wm'][normalized_key]
                    c = self._server_control['wm'][normalized_key]
                    correction = self._scaffold_lr * (c - c_i)
                    wm_state[key].add_(correction)

            for key in task_state.keys():
                normalized_key = self._normalize_state_dict_key(key)
                if normalized_key in self._client_control['task_behavior'] and normalized_key in self._server_control[
                    'task_behavior']:
                    c_i = self._client_control['task_behavior'][normalized_key]
                    c = self._server_control['task_behavior'][normalized_key]
                    correction = self._scaffold_lr * (c - c_i)
                    task_state[key].add_(correction)

    # ==================== FedSVRPG 方法 ====================
    def _get_current_gradients(self) -> Dict:
        gradients = {'wm': {}, 'task_behavior': {}}
        for name, param in self._task_behavior.named_parameters():
            if param.requires_grad and param.grad is not None:
                normalized_name = self._normalize_state_dict_key(name)
                gradients['task_behavior'][normalized_name] = param.grad.detach().clone()
        return gradients

    def _compute_vr_correction(self, current_grads: Dict) -> Dict:
        correction = {'task_behavior': {}}
        for name, grad in current_grads['task_behavior'].items():
            if name in self._reference_gradient['task_behavior']:
                ref_grad = self._reference_gradient['task_behavior'][name]
                correction['task_behavior'][name] = -(grad - ref_grad)
        return correction

    def _apply_vr_correction(self, correction: Dict):
        with torch.no_grad():
            task_state = self._task_behavior.state_dict()
            for key in task_state.keys():
                normalized_key = self._normalize_state_dict_key(key)
                if normalized_key in correction['task_behavior']:
                    corr = correction['task_behavior'][normalized_key]
                    task_state[key].add_(0.01 * corr)

    def _compute_gradient_norm(self, gradients: Dict) -> float:
        norm = 0.0
        for grad in gradients.get('task_behavior', {}).values():
            norm += torch.sum(grad ** 2).item()
        return np.sqrt(norm)

    # ==================== FedPAW-ICP 方法 ====================
    def _compute_env_complexity(self, post, context) -> float:
        pass

    def _compute_repr_diversity(self) -> float:
        pass

    def _normalize_state_dict_key(self, key: str) -> str:
        """移除 _orig_mod. 前缀"""
        if key.startswith('_orig_mod.'):
            return key[len('_orig_mod.'):]
        return key

    def _get_model_params(self) -> Dict:
        params = {'wm': {}, 'task_behavior': {}}
        wm_state = self._wm.state_dict()
        for key, value in wm_state.items():
            normalized_key = self._normalize_state_dict_key(key)
            params['wm'][normalized_key] = value.detach().clone()
        task_state = self._task_behavior.state_dict()
        for key, value in task_state.items():
            normalized_key = self._normalize_state_dict_key(key)
            params['task_behavior'][normalized_key] = value.detach().clone()
        return params

    def _compute_control_norm(self, control: Dict) -> float:
        norm = 0.0
        for name, c in control['wm'].items():
            norm += torch.sum(c ** 2).item()
        for name, c in control['task_behavior'].items():
            norm += torch.sum(c ** 2).item()
        return np.sqrt(norm)

    def _collect_dynamics_stats(self, post, context):
        pass

    def _compute_kl_divergence(self, p, q):
        try:
            if hasattr(p, 'mean') and hasattr(q, 'mean'):
                kl = torch.distributions.kl_divergence(
                    torch.distributions.Normal(p['mean'], p['std']),
                    torch.distributions.Normal(q['mean'], q['std'])
                ).mean()
                kl = float(kl.item() if hasattr(kl, 'item') else kl)
            else:
                kl = 0.0
        except Exception as e:
            kl = 0.0
        return kl

    def _init_control_variables(self) -> Dict:
        control = {'wm': {}, 'task_behavior': {}}
        wm_state = self._wm.state_dict()
        for key in wm_state.keys():
            normalized_key = self._normalize_state_dict_key(key)
            control['wm'][normalized_key] = torch.zeros_like(wm_state[key])
        task_state = self._task_behavior.state_dict()
        for key in task_state.keys():
            normalized_key = self._normalize_state_dict_key(key)
            control['task_behavior'][normalized_key] = torch.zeros_like(task_state[key])
        return control

    def _get_encoder_stats(self):
        wm = self._wm
        if hasattr(wm, 'get_buffered_erank'):
            erank = wm.get_buffered_erank()
        else:
            erank = getattr(wm, '_enc_rank_ema', 1.0)
        entropy = getattr(wm, '_enc_entropy_ema', 1.0)
        return {
            "erank": float(erank),
            "entropy": float(entropy),
        }

    def _get_task_losses(self):
        losses = {}
        for task_name in ['reward', 'image', 'cont']:
            if len(self._task_losses[task_name]) > 0:
                losses[task_name] = float(np.mean(self._task_losses[task_name][-100:]))
            else:
                losses[task_name] = 1.0
        return losses

    def _compute_new_client_control_scaffold(self) -> Dict:
        new_control = {'wm': {}, 'task_behavior': {}}
        current_params = self._get_model_params()
        K = max(1, self._local_train_steps)

        # ===== World Model =====
        for key in self._params_before_train['wm'].keys():
            theta_old = self._params_before_train['wm'][key]
            theta_new = current_params['wm'][key]

            param_delta = (theta_old - theta_new) / K

            if self._client_control is not None and key in self._client_control['wm']:
                c_i = self._client_control['wm'][key]
                c = self._server_control['wm'][key]
                new_control['wm'][key] = c_i - c + param_delta / (self._scaffold_lr + 1e-8)
            else:
                new_control['wm'][key] = param_delta / (self._scaffold_lr + 1e-8)

        # ===== Task Behavior =====
        for key in self._params_before_train['task_behavior'].keys():
            theta_old = self._params_before_train['task_behavior'][key]
            theta_new = current_params['task_behavior'][key]

            param_delta = (theta_old - theta_new) / K

            if self._client_control is not None and key in self._client_control['task_behavior']:
                c_i = self._client_control['task_behavior'][key]
                c = self._server_control['task_behavior'][key]
                new_control['task_behavior'][key] = c_i - c + param_delta / (self._scaffold_lr + 1e-8)
            else:
                new_control['task_behavior'][key] = param_delta / (self._scaffold_lr + 1e-8)

        return new_control

    def _compute_new_client_control_moswa(self) -> Dict:
        new_control = {'wm': {}, 'task_behavior': {}}
        current_params = self._get_model_params()
        avg_lr = (self._initial_lr + self._initial_lr * self._lr_decay_ratio) / 2
        sum_lr = avg_lr * self._local_iterations
        if sum_lr == 0:
            sum_lr = 1e-6

        for key in self._params_before_train['wm'].keys():
            theta_old = self._params_before_train['wm'][key]
            theta_new = current_params['wm'][key]
            param_diff = (theta_old - theta_new) / sum_lr
            if self._client_control is not None and key in self._client_control['wm']:
                c_i = self._client_control['wm'][key]
                m = self._server_control['wm'][key]
                new_control['wm'][key] = c_i - m + param_diff
            else:
                new_control['wm'][key] = param_diff

        for key in self._params_before_train['task_behavior'].keys():
            theta_old = self._params_before_train['task_behavior'][key]
            theta_new = current_params['task_behavior'][key]
            param_diff = (theta_old - theta_new) / sum_lr
            if self._client_control is not None and key in self._client_control['task_behavior']:
                c_i = self._client_control['task_behavior'][key]
                m = self._server_control['task_behavior'][key]
                new_control['task_behavior'][key] = c_i - m + param_diff
            else:
                new_control['task_behavior'][key] = param_diff

        return new_control

    def _compute_control_delta(self, new_control: Dict, old_control: Dict) -> Dict:
        delta = {'wm': {}, 'task_behavior': {}}
        for name in new_control['wm'].keys():
            delta['wm'][name] = new_control['wm'][name] - old_control['wm'][name]
        for name in new_control['task_behavior'].keys():
            delta['task_behavior'][name] = new_control['task_behavior'][name] - old_control['task_behavior'][name]
        return delta


    def get_fed_state(self) -> Dict:
        state = {
            'step': self._step,
            'update_count': self._update_count,
            'local_steps': self._local_steps,
            'global_round': self._global_round,
            'dynamics_stats_ema': self._dynamics_stats_ema,
            'task_losses': self._task_losses,
        }
        # 保存特定算法状态
        if self._config.fed_aggregation == 'scaffold':
            state['client_control'] = self._client_control
            state['server_control'] = self._server_control
        elif self._config.fed_aggregation == 'fedsvrpg':
            state['reference_policy'] = self._reference_policy
            state['reference_gradient'] = self._reference_gradient
        elif self._config.fed_aggregation == 'fedhapg':
            state['global_policy_params'] = self._global_policy_params
        elif self._use_pawid:
            state['env_complexity_buffer'] = self._env_complexity_buffer
            state['repr_diversity_buffer'] = self._repr_diversity_buffer
        elif self._use_fedmoswa:
            state['client_control'] = self._client_control
            state['server_control'] = self._server_control

        return state


    def load_fed_state(self, state: Dict):
        self._step = state.get('step', 0)
        self._update_count = state.get('update_count', 0)
        self._local_steps = state.get('local_steps', 0)
        self._global_round = state.get('global_round', 0)
        self._dynamics_stats_ema = state.get('dynamics_stats_ema', self._dynamics_stats_ema)
        self._task_losses = state.get('task_losses', self._task_losses)

        if self._config.fed_aggregation == 'scaffold':
            self._client_control = state.get('client_control')
            self._server_control = state.get('server_control')
        elif self._config.fed_aggregation == 'fedsvrpg':
            self._reference_policy = state.get('reference_policy')
            self._reference_gradient = state.get('reference_gradient')
        elif self._config.fed_aggregation == 'fedhapg':
            self._global_policy_params = state.get('global_policy_params')
        elif self._use_pawid:
            self._env_complexity_buffer = state.get('env_complexity_buffer', [])
            self._repr_diversity_buffer = state.get('repr_diversity_buffer', [])
        elif self._use_fedmoswa:
            self._client_control = state.get('client_control')
            self._server_control = state.get('server_control')

    def get_weights(self):
        wm_state = self._wm.state_dict()
        task_state = self._task_behavior.state_dict()

        weights = {
            'wm': wm_state,
            'task_behavior': task_state,
            'local_steps': self._local_steps,
            'dynamics_stats': self._dynamics_stats_ema.copy(),
            'encoder_stats': self._get_encoder_stats(),
            'task_losses': self._get_task_losses(),
        }

        if self._config.fed_aggregation == 'scaffold' and self._params_before_train is not None:
            new_client_control = self._compute_new_client_control_scaffold()
            if self._client_control is not None:
                delta_c = self._compute_control_delta(new_client_control, self._client_control)
            else:
                delta_c = new_client_control
            weights['scaffold_delta_c'] = delta_c
            weights['scaffold_local_steps'] = self._local_train_steps
            self._client_control = new_client_control

        if self._config.fed_aggregation == 'fedsvrpg':
            weights['svrpg_gradients'] = self._get_current_gradients()

        if self._config.fed_aggregation == 'fedhapg':
            policy_params = {}
            for name, param in self._task_behavior.actor.named_parameters():
                policy_params[name] = param.detach().clone()
            weights['hapg_policy_params'] = policy_params

        if self._use_pawid:
            if len(self._env_complexity_buffer) > 0:
                avg_complexity = float(np.mean(self._env_complexity_buffer[-50:]))
                weights['pawid_env_complexity'] = avg_complexity
            else:
                weights['pawid_env_complexity'] = 1.0
            if len(self._repr_diversity_buffer) > 0:
                avg_diversity = float(np.mean(self._repr_diversity_buffer))
                weights['pawid_repr_diversity'] = avg_diversity
            else:
                weights['pawid_repr_diversity'] = 1.0

        if self._use_fedmoswa and self._params_before_train is not None:
            new_client_control = self._compute_new_client_control_moswa()
            if self._client_control is not None:
                delta_c = self._compute_control_delta(new_client_control, self._client_control)
            else:
                delta_c = new_client_control
            weights['moswa_delta_c'] = delta_c
            weights['moswa_local_iterations'] = self._local_iterations
            self._client_control = new_client_control

        return weights

    def set_weights(self, weights):
        self._wm.load_state_dict(weights['wm'])
        self._task_behavior.load_state_dict(weights['task_behavior'])

        if self._config.fed_aggregation == 'fedprox':
            self._global_params = {'wm': {}, 'task_behavior': {}}
            for name, param in self._wm.named_parameters():
                self._global_params['wm'][name] = param.detach().clone()
            for name, param in self._task_behavior.named_parameters():
                self._global_params['task_behavior'][name] = param.detach().clone()

        elif self._config.fed_aggregation == 'scaffold':
            if 'scaffold_server_control' in weights:
                self._server_control = weights['scaffold_server_control']
            else:
                self._server_control = self._init_control_variables()
            if self._client_control is None:
                self._client_control = self._init_control_variables()
            self._local_train_steps = 0
            self._params_before_train = None

        elif self._config.fed_aggregation == 'fedsvrpg':
            if 'svrpg_reference_policy' in weights:
                self._reference_policy = weights['svrpg_reference_policy']
            if 'svrpg_reference_gradient' in weights:
                self._reference_gradient = weights['svrpg_reference_gradient']

        elif self._config.fed_aggregation == 'fedhapg' and 'hapg_global_policy' in weights:
            self._global_policy_params = weights['hapg_global_policy']

        elif self._use_pawid:
            if len(self._env_complexity_buffer) > 20:
                self._env_complexity_buffer = self._env_complexity_buffer[-20:]
            if len(self._repr_diversity_buffer) > 5:
                self._repr_diversity_buffer = self._repr_diversity_buffer[-5:]
            self._latent_features_buffer = []

        elif self._use_fedmoswa:
            if 'moswa_server_control' in weights:
                self._server_control = weights['moswa_server_control']
            else:
                self._server_control = self._init_control_variables()
            if self._client_control is None:
                self._client_control = self._init_control_variables()

        elif self._use_sam_fmgt:
            self._current_global_params = self._deep_copy_weights(weights)

            self._global_trajectory.append(self._current_global_params)

            if len(self._global_trajectory) > self._trajectory_max_len:
                self._global_trajectory.pop(0)

            print(f"Client {self._client_id}: Trajectory size = {len(self._global_trajectory)}")

        self._global_round += 1
        print(f"Client {self._client_id}: Updated to global round {self._global_round}")

    def _deep_copy_weights(self, weights: Dict) -> Dict:
        copy_weights = {'wm': {}, 'task_behavior': {}}
        for key in weights['wm'].keys():
            copy_weights['wm'][key] = weights['wm'][key].clone().detach()
        for key in weights['task_behavior'].keys():
            copy_weights['task_behavior'][key] = weights['task_behavior'][key].clone().detach()
        return copy_weights


class Client:

    def __init__(self, config, client_id):
        self.config = config
        self.idx = client_id
        self.device = config.device
        self.local_steps = config.local_steps
        self.eval_rewards = []

        self.save_warmup_model = tools.Once()
        self.load_warmup_model = tools.Once()

        self.seed = config.seed + self.idx
        random.seed(self.seed)

        self.prefill = config.prefill

        self.logdir = pathlib.Path(config.logdir).expanduser() / f"client_{self.idx}"
        self.traindir = self.logdir / "train_eps"
        self.evaldir = self.logdir / "eval_eps"

        self.logdir.mkdir(parents=True, exist_ok=True)
        self.traindir.mkdir(parents=True, exist_ok=True)
        self.evaldir.mkdir(parents=True, exist_ok=True)

        self.step = self._count_steps(self.traindir)
        self.logger = tools.Logger(self.logdir, config.action_repeat * self.step)

        self.train_eps = tools.load_episodes(self.traindir, limit=config.dataset_size)
        self.eval_eps = tools.load_episodes(self.evaldir, limit=1)

        self._create_envs()

        self.state = None
        self._prefill_dataset()

        self.train_dataset = self._make_dataset(self.train_eps)
        self.eval_dataset = self._make_dataset(self.eval_eps)

        self.agent = FederatedDreamer(
            self.train_envs[0].observation_space,
            self.train_envs[0].action_space,
            self.config,
            self.logger,
            self.train_dataset,
            client_id=self.idx
        )

        self._try_load_latest_checkpoint()

        print(f'Client {self.idx} initialized (seed={self.seed})')

    def _try_load_latest_checkpoint(self):
        latest_path = self.logdir / "latest.pt"
        if latest_path.exists():
            try:
                checkpoint = torch.load(latest_path, weights_only=False)

                state_dict = checkpoint["agent_state_dict"]
                if "_extra_state" in state_dict:
                    del state_dict["_extra_state"]

                self.agent.load_state_dict(state_dict)
                tools.recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
                if "extra_state" in checkpoint:
                    self.agent.load_fed_state(checkpoint["extra_state"])

                self.agent._should_pretrain._once = False
                self.save_warmup_model._once = False
                self.load_warmup_model._once = False
                print(f"Client {self.idx}: Resumed from latest checkpoint.")
            except Exception as e:
                print(f"Client {self.idx}: Failed to load latest checkpoint: {e}")

    def _count_steps(self, folder):
        return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))

    def _make_dataset(self, episodes):
        generator = tools.sample_episodes(episodes, self.config.batch_length)
        dataset = tools.from_generator(generator, self.config.batch_size)
        return dataset

    def _create_envs(self):
        make = lambda mode, id: self._make_env(mode, id, no_iid=self.config.no_iid, alpha=self.config.alpha_dirichlet)

        self.train_envs = [make("train", i) for i in range(self.config.envs)]
        self.eval_envs = [make("eval", i) for i in range(self.config.envs)]

        if self.config.parallel:
            self.train_envs = [Parallel(env, "process") for env in self.train_envs]
            self.eval_envs = [Parallel(env, "process") for env in self.eval_envs]
        else:
            self.train_envs = [Damy(env) for env in self.train_envs]
            self.eval_envs = [Damy(env) for env in self.eval_envs]

        self.acts = self.train_envs[0].action_space
        self.config.num_actions = self.acts.n if hasattr(self.acts, "n") else self.acts.shape[0]

    def _make_env(self, mode, id, no_iid=True, alpha=0.1):
        suite, task = self.config.task.split("_", 1)
        seed = self.seed + self.idx

        if suite == "dmc":
            import envs.dmc as dmc
            env = dmc.DeepMindControl(
                task, self.config.action_repeat, self.config.size, seed=seed
            )
            env = wrappers.NormalizeActions(env)
        elif suite == "atari":
            import envs.atari as atari

            mode = None
            difficulty = None
            if no_iid:
                modes, difficulties = tools.get_modde_difficulty(task, seed=seed)
                mode = tools.ordered_dirichlet_selection(modes, id, alpha=alpha, seed=seed + 100)
                difficulty = tools.ordered_dirichlet_selection(difficulties, id, alpha=alpha, seed=seed + 50)

            with open(os.path.join(self.logdir, 'game.txt'), 'a', encoding='utf-8') as f:
                f.write(f'{task} difficulty: {difficulty}\n')
                f.write(f'{task} mode: {mode}\n')

            env = atari.Atari(
                task,
                self.config.action_repeat,
                self.config.size,
                gray=self.config.grayscale,
                noops=self.config.noops,
                lives=self.config.lives,
                sticky=self.config.stickey,
                actions=self.config.actions,
                resize=self.config.resize,
                seed=seed,
                mode=mode,
                difficulty=difficulty,
            )
            env = wrappers.OneHotAction(env)
        elif suite == "crafter":
            import envs.crafter as crafter
            env = crafter.Crafter(task, self.config.size, seed=seed)
            env = wrappers.OneHotAction(env)
        else:
            raise NotImplementedError(suite)

        env = wrappers.TimeLimit(env, self.config.time_limit)
        env = wrappers.SelectAction(env, key="action")
        env = wrappers.UUID(env)
        return env

    def _prefill_dataset(self):
        if os.listdir(self.traindir):
            self.prefill = 0
            return

        prefill = max(0, self.prefill - self._count_steps(self.traindir))
        if prefill == 0:
            return

        print(f"Client {self.idx}: Prefill dataset ({prefill} steps).")

        if hasattr(self.acts, "discrete"):
            random_actor = tools.OneHotDist(
                torch.zeros(self.config.num_actions).repeat(self.config.envs, 1)
            )
        else:
            random_actor = torchd.independent.Independent(
                torchd.uniform.Uniform(
                    torch.tensor(self.acts.low).repeat(self.config.envs, 1),
                    torch.tensor(self.acts.high).repeat(self.config.envs, 1),
                ),
                1,
            )

        def random_agent(o, d, s):
            action = random_actor.sample()
            logprob = random_actor.log_prob(action)
            return {"action": action, "logprob": logprob}, None

        self.state = tools.simulate(
            random_agent,
            self.train_envs,
            self.train_eps,
            self.traindir,
            self.logger,
            limit=self.config.dataset_size,
            steps=prefill,
        )
        self.logger.step += prefill * self.config.action_repeat

    def train(self):
        start = time.time()
        print('=' * 20, f'Client {self.idx}', '=' * 20)

        self.agent._dynamics_stats_ema = {
            "pred_consistency": 0.0,
            "latent_health": 0.0,
            "env_complexity": 0.0,
        }
        self.agent._dynamics_stats_count = 0
        self.agent._task_losses = {k: [] for k in ['reward', 'image', 'cont']}

        self.agent._local_iterations = 0

        self.logger.write()
        self.agent.to(self.device)

        if (self.logdir / "warmup.pt").exists() and self.load_warmup_model():
            checkpoint = torch.load(self.logdir / "warmup.pt", weights_only=False)

            state_dict = checkpoint["agent_state_dict"]
            if "_extra_state" in state_dict:
                del state_dict["_extra_state"]

            self.agent.load_state_dict(state_dict)
            tools.recursively_load_optim_state_dict(self.agent, checkpoint["optims_state_dict"])
            self.agent._should_pretrain._once = False
            self.save_warmup_model._once = False
            print(f"Client {self.idx}: Warmup model loaded.")

        self.agent.train()

        self.state = tools.simulate(
            self.agent,
            self.train_envs,
            self.train_eps,
            self.traindir,
            self.logger,
            limit=self.config.dataset_size,
            steps=self.local_steps,
            state=self.state,
        )

        items_to_save = {
            "agent_state_dict": self.agent.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(self.agent),
            "extra_state": self.agent.get_fed_state(),
        }
        torch.save(items_to_save, self.logdir / "latest.pt")

        if self.save_warmup_model():
            torch.save(items_to_save, self.logdir / "warmup.pt")
            print(f"Client {self.idx}: Warmup model saved.")

        print(f'Client {self.idx} train time: {time.time() - start:.2f}s')

    def evaluate(self, record_reward=True):
        self.logger.write()
        self.agent.to(self.device)
        self.agent.eval()

        eval_policy = functools.partial(self.agent, training=False)
        eval_reward = tools.simulate(
            eval_policy,
            self.eval_envs,
            self.eval_eps,
            self.evaldir,
            self.logger,
            is_eval=True,
            episodes=self.config.eval_episode_num,
        )[-1]

        # for env in eval_envs:
        #     env.close()

        if record_reward:
            self.eval_rewards.append(eval_reward)

        if self.config.video_pred_log:
            video_pred = self.agent._wm.video_pred(next(self.eval_dataset))
            self.logger.video("eval_openl", to_np(video_pred))

        return eval_reward

    def close_envs(self):
        for env in self.train_envs + self.eval_envs:
            try:
                env.close()
            except Exception:
                pass


class FederatedServer:

    def __init__(self, config):
        self.config = config
        self.num_clients = config.num_clients
        self.global_rounds = config.global_rounds
        self.select_ratio = config.select_ratio
        self.select_num_clients = max(1, int(self.select_ratio * self.num_clients))
        self.eval_gap = config.eval_gap
        self.fed_aggregation = config.fed_aggregation
        self.device = config.device
        self.logdir = pathlib.Path(config.logdir).expanduser()

        self.clients = []
        self.selected_clients = []
        self.client_weights = []

        self.server_logdir = self.logdir / "Server"
        self.server_logdir.mkdir(parents=True, exist_ok=True)
        self.logger = tools.Logger(self.server_logdir, 0)

        self.start_round = 0

        self.global_weights = None

        self._initialize_clients()

        self.task_weights = {
            'reward': 0.4,
            'image': 0.4,
            'cont': 0.2,
        }
        total = sum(self.task_weights.values())
        self.task_weights = {k: v / total for k, v in self.task_weights.items()}

        self._init_method_specific_params()

        self._print_server_config()

        self.load_checkpoint()

        # SAM-FMGT
        self.samfmgt_momentum = getattr(self.config, 'samfmgt_momentum', 0.9)
        self.samfmgt_velocity = None  # 动量缓存

    def _init_method_specific_params(self):
        # SCAFFOLD
        self.scaffold_lr = getattr(self.config, 'scaffold_lr', 1e-3)
        self.server_control = None

        # FedSVRPG
        self.svrpg_reference_gradient = None

        # FedHAPG
        self.hapg_lambda = getattr(self.config, 'hapg_lambda', 0.5)

        self.pawid_tau = getattr(self.config, 'pawid_tau', 1.0)
        self.pawid_b = getattr(self.config, 'pawid_b', 0.6)
        self.pawid_beta = getattr(self.config, 'pawid_beta', 0.5)
        self.pawid_epsilon_calib = 1e-5
        self.prev_global_weights = None

        # FedSWA
        self.swa_alpha = getattr(self.config, 'swa_alpha', 0.5)

        # FedMoSWA
        self.moswa_gamma = getattr(self.config, 'moswa_gamma', 0.9)

    def _print_server_config(self):
        print(f"\n{'=' * 60}")
        print(f"Federated Server Initialized")
        print(f"{'=' * 60}")
        print(f"Total clients: {self.num_clients}")
        print(f"Selected per round: {self.select_num_clients} ({self.select_ratio:.1%})")
        print(f"Aggregation method: {self.fed_aggregation.upper()}")
        print(f"Global rounds: {self.global_rounds}")
        print(f"Evaluation gap: {self.eval_gap}")

        if self.fed_aggregation == 'scaffold':
            print(f"SCAFFOLD lr: {self.scaffold_lr}")
        elif self.fed_aggregation == 'fedhapg':
            print(f"FedHAPG λ: {self.hapg_lambda}")
        elif self.fed_aggregation == 'fedpawid':
            print(f"FedPAW-ICP τ: {self.pawid_tau}")
            print(f"FedPAW-ICP λ_comp: {self.pawid_lambda_comp}")
            print(f"FedPAW-ICP λ_rep: {self.pawid_lambda_rep}")
            print(f"FedPAW-ICP β: {self.pawid_beta}")
        elif self.fed_aggregation == 'fedswa':
            print(f"FedSWA α: {self.swa_alpha}")
        elif self.fed_aggregation == 'fedmoswa':
            print(f"FedMoSWA γ: {self.moswa_gamma}")
            print(f"FedMoSWA α: {self.swa_alpha}")

        print(f"{'=' * 60}\n")

    def _initialize_clients(self):
        print("Initializing clients...")
        for i in range(self.num_clients):
            client = Client(self.config, client_id=i)
            self.clients.append(client)
        print(f"All {self.num_clients} clients initialized.\n")

    def select_clients(self):
        self.selected_clients = random.sample(self.clients, self.select_num_clients)
        selected_ids = [c.idx for c in self.selected_clients]
        print(f"Selected clients: {selected_ids}")

    def receive_models(self):
        self.client_weights = []
        for client in self.selected_clients:
            weights = client.agent.get_weights()
            self.client_weights.append(weights)

    def save_checkpoint(self, round_num):
        checkpoint_path = self.server_logdir / "server_checkpoint.pt"
        state = {
            'round': round_num,
            'global_weights': self.global_weights,
            'server_control': self.server_control,
            'svrpg_reference_gradient': self.svrpg_reference_gradient,
            'prev_global_weights': self.prev_global_weights,
        }
        torch.save(state, checkpoint_path)
        torch.save(state, self.server_logdir / "server_checkpoint_backup.pt")
        print(f"Server checkpoint saved for round {round_num}")

    def load_checkpoint(self):
        checkpoint_path = self.server_logdir / "server_checkpoint.pt"
        if not checkpoint_path.exists():
            checkpoint_path = self.server_logdir / "server_checkpoint_backup.pt"

        if checkpoint_path.exists():
            try:
                print(f"Loading server checkpoint from {checkpoint_path}...")
                state = torch.load(checkpoint_path, weights_only=False)
                self.start_round = state['round']
                self.global_weights = state['global_weights']
                self.server_control = state['server_control']
                self.svrpg_reference_gradient = state['svrpg_reference_gradient']
                self.prev_global_weights = state['prev_global_weights']

                self.logger.step = self.start_round
                print(f"Resuming from Round {self.start_round}")
            except Exception as e:
                print(f"Failed to load server checkpoint: {e}")
                print("Starting from scratch.")
        else:
            print("No checkpoint found. Starting training from scratch.")

    def aggregate_models(self):
        print(f"\nAggregating models with {self.fed_aggregation}...")

        method_map = {
            'fedavg': self._fedavg,
            'fedprox': self._fedavg,
            'scaffold': self._scaffold_aggregation,
            'fedsvrpg': self._fedsvrpg_aggregation,
            'fedhapg': self._fedhapg_aggregation,
            'fedpawid': self._fedpawid_aggregation,
            'fedswa': self._fedswa_aggregation,
            'fedmoswa': self._fedmoswa_aggregation,
            'fedwm': self._fedwm_aggregation,
            'samfmgt': self._samfmgt_aggregation,
            'local': self._local,
        }

        aggregation_func = method_map.get(self.fed_aggregation, self._fedavg)
        aggregation_func()

    def _local(self):
        return

    def _samfmgt_aggregation(self):

        print("=" * 80)
        print("SAM-FMGT: Sharpness-Aware Minimization + Global Trajectory")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        aggregated = {'wm': {}, 'task_behavior': {}}

        for key in self.client_weights[0]['wm'].keys():
            aggregated['wm'][key] = sum(
                client['wm'][key] for client in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            aggregated['task_behavior'][key] = sum(
                client['task_behavior'][key] for client in self.client_weights
            ) / num_clients

        if self.global_weights is not None:
            if self.samfmgt_velocity is None:
                self.samfmgt_velocity = {'wm': {}, 'task_behavior': {}}
                for key in aggregated['wm'].keys():
                    self.samfmgt_velocity['wm'][key] = torch.zeros_like(aggregated['wm'][key])
                for key in aggregated['task_behavior'].keys():
                    self.samfmgt_velocity['task_behavior'][key] = torch.zeros_like(aggregated['task_behavior'][key])

            smoothed = {'wm': {}, 'task_behavior': {}}

            for key in aggregated['wm'].keys():
                delta = aggregated['wm'][key] - self.global_weights['wm'][key]
                self.samfmgt_velocity['wm'][key] = (
                        self.samfmgt_momentum * self.samfmgt_velocity['wm'][key] + delta
                )
                smoothed['wm'][key] = self.global_weights['wm'][key] + self.samfmgt_velocity['wm'][key]

            for key in aggregated['task_behavior'].keys():
                delta = aggregated['task_behavior'][key] - self.global_weights['task_behavior'][key]
                self.samfmgt_velocity['task_behavior'][key] = (
                        self.samfmgt_momentum * self.samfmgt_velocity['task_behavior'][key] + delta
                )
                smoothed['task_behavior'][key] = (
                        self.global_weights['task_behavior'][key] + self.samfmgt_velocity['task_behavior'][key]
                )

            aggregated = smoothed

        if self.global_weights is not None:
            velocity_norm = self._compute_velocity_norm()
            print(f"Velocity norm: {velocity_norm:.6f}")
            self.logger.scalar("samfmgt/velocity_norm", float(velocity_norm))

        self.global_weights = aggregated
        print("SAM-FMGT aggregation completed.\n")

    def _compute_velocity_norm(self) -> float:
        if self.samfmgt_velocity is None:
            return 0.0

        norm = 0.0
        for v in self.samfmgt_velocity['wm'].values():
            norm += torch.sum(v ** 2).item()
        for v in self.samfmgt_velocity['task_behavior'].values():
            norm += torch.sum(v ** 2).item()
        return np.sqrt(norm)

    # ==================== FedAvg ====================
    def _fedavg(self):
        num_clients = len(self.selected_clients)
        aggregated = {}

        aggregated['wm'] = {}
        for key in self.client_weights[0]['wm'].keys():
            aggregated['wm'][key] = sum(
                client['wm'][key] for client in self.client_weights
            ) / num_clients

        aggregated['task_behavior'] = {}
        for key in self.client_weights[0]['task_behavior'].keys():
            aggregated['task_behavior'][key] = sum(
                client['task_behavior'][key] for client in self.client_weights
            ) / num_clients

        self.global_weights = aggregated
        print("FedAvg aggregation completed.")

    # ==================== SCAFFOLD ====================
    def _scaffold_aggregation(self):
        print("=" * 80)
        print("SCAFFOLD Aggregation")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        if self.server_control is None:
            print("Initializing server control variables...")
            self.server_control = self._init_server_control()

        aggregated = {'wm': {}, 'task_behavior': {}}

        for key in self.client_weights[0]['wm'].keys():
            aggregated['wm'][key] = sum(
                client['wm'][key] for client in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            aggregated['task_behavior'][key] = sum(
                client['task_behavior'][key] for client in self.client_weights
            ) / num_clients

        if all('scaffold_delta_c' in w for w in self.client_weights):
            print("Updating server control variables...")
            self._update_server_control()

        aggregated['scaffold_server_control'] = self.server_control
        self.global_weights = aggregated

        self._log_scaffold_debug()
        print("SCAFFOLD aggregation completed.\n")

    def _init_server_control(self) -> Dict:
        control = {'wm': {}, 'task_behavior': {}}
        first_client = self.client_weights[0]

        for key in first_client['wm'].keys():
            normalized_key = self._normalize_state_dict_key(key)
            control['wm'][normalized_key] = torch.zeros_like(first_client['wm'][key])

        for key in first_client['task_behavior'].keys():
            normalized_key = self._normalize_state_dict_key(key)
            control['task_behavior'][normalized_key] = torch.zeros_like(first_client['task_behavior'][key])

        return control

    def _update_server_control(self):
        num_clients = len(self.selected_clients)
        max_update_norm = 1.0

        old_norm = self._compute_control_norm(self.server_control)

        # World Model
        for key in self.server_control['wm'].keys():
            delta_sum = torch.zeros_like(self.server_control['wm'][key])
            valid_count = 0

            for w in self.client_weights:
                if 'scaffold_delta_c' in w and key in w['scaffold_delta_c'].get('wm', {}):
                    delta_sum = delta_sum + w['scaffold_delta_c']['wm'][key]
                    valid_count += 1

            if valid_count > 0:
                update = delta_sum / valid_count

                update_norm = torch.norm(update).item()
                if update_norm > max_update_norm:
                    update = update * (max_update_norm / (update_norm + 1e-8))
                    print(f"⚠️  Clipping WM {key}: norm {update_norm:.4f} → {max_update_norm}")

                self.server_control['wm'][key] = self.server_control['wm'][key] + update

        # Task Behavior
        for key in self.server_control['task_behavior'].keys():
            delta_sum = torch.zeros_like(self.server_control['task_behavior'][key])
            valid_count = 0

            for w in self.client_weights:
                if 'scaffold_delta_c' in w and key in w['scaffold_delta_c'].get('task_behavior', {}):
                    delta_sum = delta_sum + w['scaffold_delta_c']['task_behavior'][key]
                    valid_count += 1

            if valid_count > 0:
                update = delta_sum / valid_count
                update_norm = torch.norm(update).item()
                if update_norm > max_update_norm:
                    update = update * (max_update_norm / (update_norm + 1e-8))
                    print(f"⚠️  Clipping task {key}: norm {update_norm:.4f} → {max_update_norm}")

                self.server_control['task_behavior'][key] = self.server_control['task_behavior'][key] + update

        new_norm = self._compute_control_norm(self.server_control)
        growth = (new_norm - old_norm) / (old_norm + 1e-8)

        print(f"Server Control: {old_norm:.4f} → {new_norm:.4f} (Δ={growth:+.2%})")
        self.logger.scalar("scaffold/server_control_norm", float(new_norm))

    def _log_scaffold_debug(self):
        print("\n" + "=" * 80)
        print("SCAFFOLD Control Variables Statistics")
        print("=" * 80)

        server_control_norm = self._compute_control_norm(self.server_control)
        print(f"Server Control Norm: {server_control_norm:.6f}")

        print(f"\n{'Client':<8} {'Δc Norm':>12} {'Local Steps':>12}")
        print("-" * 80)

        for idx, client in enumerate(self.selected_clients):
            if 'scaffold_delta_c' in self.client_weights[idx]:
                delta_c = self.client_weights[idx]['scaffold_delta_c']
                delta_norm = self._compute_control_norm(delta_c)
                local_steps = self.client_weights[idx].get('scaffold_local_steps', 0)
                print(f"{client.idx:<8d} {delta_norm:>12.6f} {local_steps:>12d}")

        print("=" * 80)
        self.logger.scalar("scaffold/server_control_norm", float(server_control_norm))
        print()

    # ==================== FedSVRPG ====================
    def _fedsvrpg_aggregation(self):
        print("=" * 80)
        print("FedSVRPG-M Aggregation")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        aggregated = {'wm': {}, 'task_behavior': {}}
        for key in self.client_weights[0]['wm'].keys():
            aggregated['wm'][key] = sum(
                c['wm'][key] for c in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            aggregated['task_behavior'][key] = sum(
                c['task_behavior'][key] for c in self.client_weights
            ) / num_clients

        if all('svrpg_gradients' in w for w in self.client_weights):
            agg_grad = {'task_behavior': {}}
            first_grad = self.client_weights[0]['svrpg_gradients']

            for key in first_grad['task_behavior'].keys():
                grads = [w['svrpg_gradients']['task_behavior'][key]
                         for w in self.client_weights
                         if 'svrpg_gradients' in w]
                agg_grad['task_behavior'][key] = sum(grads) / len(grads)

            aggregated['svrpg_reference_gradient'] = agg_grad
            self.svrpg_reference_gradient = agg_grad

        self.global_weights = aggregated
        print("FedSVRPG-M aggregation completed.\n")

    # ==================== FedHAPG ====================
    def _fedhapg_aggregation(self):
        print("=" * 80)
        print("FedHAPG-M Aggregation")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        aggregated = {'wm': {}, 'task_behavior': {}}
        for key in self.client_weights[0]['wm'].keys():
            aggregated['wm'][key] = sum(
                c['wm'][key] for c in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            aggregated['task_behavior'][key] = sum(
                c['task_behavior'][key] for c in self.client_weights
            ) / num_clients

        if all('hapg_policy_params' in w for w in self.client_weights):
            global_policy = {}
            first_policy = self.client_weights[0]['hapg_policy_params']

            for key in first_policy.keys():
                params = [w['hapg_policy_params'][key]
                          for w in self.client_weights]
                global_policy[key] = sum(params) / len(params)

            aggregated['hapg_global_policy'] = global_policy

        self.global_weights = aggregated
        print("FedHAPG-M aggregation completed.\n")

    def _fedpawid_aggregation(self):
        pass

    def _compute_client_deltas(self):
        client_deltas = []
        for idx, client_weights in enumerate(self.client_weights):
            delta = {'wm': {}, 'task_behavior': {}}
            for key in client_weights['wm'].keys():
                if self.prev_global_weights is not None:
                    delta['wm'][key] = (
                            client_weights['wm'][key] -
                            self.prev_global_weights['wm'][key]
                    )
                else:
                    delta['wm'][key] = client_weights['wm'][key]
            for key in client_weights['task_behavior'].keys():
                if self.prev_global_weights is not None:
                    delta['task_behavior'][key] = (
                            client_weights['task_behavior'][key] -
                            self.prev_global_weights['task_behavior'][key]
                    )
                else:
                    delta['task_behavior'][key] = client_weights['task_behavior'][key]
            client_deltas.append(delta)
        return client_deltas

    def _compute_average_delta(self, client_deltas):
        pass

    def _compute_alignment_scores(self, client_deltas, delta_avg):
        pass

    def _weighted_aggregation(self, weights):
        pass

    def _log_fedpawid_debug(
            self,
            alignment_scores: np.ndarray,
            u_prior: np.ndarray,
            tilde_H: np.ndarray,
            tilde_R: np.ndarray,
            m_likelihood: np.ndarray,
            w_fused: np.ndarray,
            alpha_weights: np.ndarray
    ):
        print("\n" + "=" * 100)
        print("FedPAW-ICP Aggregation Details (Log-Robust Normalization)")
        print("=" * 100)
        print(f"{'Client':<8} {'Align Score':>12} {'u_prior':>12} {'tilde_H':>12} {'tilde_R':>12} "
              f"{'m_likeli':>12} {'w_fused':>12} {'alpha':>12} {'Contrib%':>12}")
        print("-" * 100)

        for idx, client in enumerate(self.selected_clients):
            contrib = alpha_weights[idx] * 100
            print(f"{client.idx:<8d} "
                  f"{alignment_scores[idx]:>12.6f} "
                  f"{u_prior[idx]:>12.6f} "
                  f"{tilde_H[idx]:>12.6f} "
                  f"{tilde_R[idx]:>12.6f} "
                  f"{m_likelihood[idx]:>12.6f} "
                  f"{w_fused[idx]:>12.6f} "
                  f"{alpha_weights[idx]:>12.6f} "
                  f"{contrib:>11.2f}%")

        print("=" * 100)
        print(f"\nStatistics:")
        print(f"  Alignment Score - Mean: {alignment_scores.mean():.4f}, Std: {alignment_scores.std():.4f}")
        print(f"  Weight Entropy: {self._compute_entropy(alpha_weights):.4f}")
        print(f"  Effective Clients: {1 / np.sum(alpha_weights ** 2):.2f}/{len(alpha_weights)}")
        print()

    # ==================== FedSWA ====================
    def _fedswa_aggregation(self):
        print("=" * 80)
        print("FedSWA: Stochastic Weight Averaging")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        v_t = {'wm': {}, 'task_behavior': {}}

        for key in self.client_weights[0]['wm'].keys():
            v_t['wm'][key] = sum(
                client['wm'][key] for client in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            v_t['task_behavior'][key] = sum(
                client['task_behavior'][key] for client in self.client_weights
            ) / num_clients

        if self.global_weights is not None:
            aggregated = self._apply_ema(v_t, self.global_weights, self.swa_alpha)
        else:
            aggregated = v_t

        self.global_weights = aggregated
        print(f"EMA aggregation with α={self.swa_alpha}")
        print("FedSWA aggregation completed.\n")

    # ==================== FedMoSWA ====================
    def _fedmoswa_aggregation(self):
        print("=" * 80)
        print("FedMoSWA: Momentum-based SWA")
        print("=" * 80)

        num_clients = len(self.selected_clients)

        if self.server_control is None:
            print("Initializing server control variables...")
            self.server_control = self._init_server_control()

        v_t = {'wm': {}, 'task_behavior': {}}
        for key in self.client_weights[0]['wm'].keys():
            v_t['wm'][key] = sum(
                client['wm'][key] for client in self.client_weights
            ) / num_clients

        for key in self.client_weights[0]['task_behavior'].keys():
            v_t['task_behavior'][key] = sum(
                client['task_behavior'][key] for client in self.client_weights
            ) / num_clients

        if self.global_weights is not None:
            aggregated = self._apply_ema(v_t, self.global_weights, self.swa_alpha)
        else:
            aggregated = v_t

        if all('moswa_delta_c' in w for w in self.client_weights):
            self._update_server_control_moswa()

        aggregated['moswa_server_control'] = self.server_control
        self.global_weights = aggregated

        self._log_moswa_debug()
        print("FedMoSWA aggregation completed.\n")

    def _apply_ema(self, v_t, theta_prev, alpha):
        updated = {'wm': {}, 'task_behavior': {}}

        for key in v_t['wm'].keys():
            if key in theta_prev['wm']:
                updated['wm'][key] = theta_prev['wm'][key] + alpha * (v_t['wm'][key] - theta_prev['wm'][key])
            else:
                updated['wm'][key] = v_t['wm'][key]

        for key in v_t['task_behavior'].keys():
            if key in theta_prev['task_behavior']:
                updated['task_behavior'][key] = theta_prev['task_behavior'][key] + alpha * (
                        v_t['task_behavior'][key] - theta_prev['task_behavior'][key])
            else:
                updated['task_behavior'][key] = v_t['task_behavior'][key]

        return updated

    def _update_server_control_moswa(self):
        num_clients = len(self.selected_clients)
        gamma = self.moswa_gamma

        for key in self.server_control['wm'].keys():
            delta_sum = sum(
                w['moswa_delta_c']['wm'][key]
                for w in self.client_weights
                if 'moswa_delta_c' in w and key in w['moswa_delta_c']['wm']
            )
            self.server_control['wm'][key] = self.server_control['wm'][key] + gamma * (delta_sum / num_clients)

        for key in self.server_control['task_behavior'].keys():
            delta_sum = sum(
                w['moswa_delta_c']['task_behavior'][key]
                for w in self.client_weights
                if 'moswa_delta_c' in w and key in w['moswa_delta_c']['task_behavior']
            )
            self.server_control['task_behavior'][key] = self.server_control['task_behavior'][key] + gamma * (
                    delta_sum / num_clients)

    def _log_moswa_debug(self):
        print("\n" + "=" * 80)
        print("FedMoSWA Control Variables Statistics")
        print("=" * 80)

        server_control_norm = self._compute_control_norm(self.server_control)
        print(f"Server Control Norm: {server_control_norm:.6f}")

        print(f"\n{'Client':<8} {'Δc Norm':>12} {'Local Iters':>12}")
        print("-" * 80)

        for idx, client in enumerate(self.selected_clients):
            if 'moswa_delta_c' in self.client_weights[idx]:
                delta_c = self.client_weights[idx]['moswa_delta_c']
                delta_norm = self._compute_control_norm(delta_c)
                local_iters = self.client_weights[idx].get('moswa_local_iterations', 0)
                print(f"{client.idx:<8d} {delta_norm:>12.6f} {local_iters:>12d}")

        print("=" * 80)
        self.logger.scalar("moswa/server_control_norm", float(server_control_norm))
        print()

    # ==================== FedWM ====================
    def _fedwm_aggregation(self):
        pass

    def _compute_dynamics_weights(self, tau_cons=1.0):
        weights = []
        for w in self.client_weights:
            stats = w['dynamics_stats']
            L_cons = stats['pred_consistency']
            w_conf = np.exp(-L_cons / tau_cons)
            S_h = stats['latent_health']
            phi_health = S_h * (1 - S_h)
            C_comp = stats['env_complexity']
            alpha = w_conf * phi_health * C_comp
            weights.append(alpha)
        total = sum(weights) + 1e-8
        weights = [w / total for w in weights]
        return weights

    def _compute_encoder_weights(self, tau_d=1.0):
        eranks = []
        for w in self.client_weights:
            enc_stats = w['encoder_stats']
            erank = enc_stats['erank']
            eranks.append(erank)
        eranks = np.array(eranks)
        exp_scores = np.exp(eranks / tau_d)
        weights = exp_scores / exp_scores.sum()
        return weights.tolist()

    def _compute_decoder_weights(self, sigma=1.0, epsilon=1e-8):
        task_losses_all = []
        for w in self.client_weights:
            task_losses_all.append(w['task_losses'])

        task_baselines = {}
        for task_name in ['reward', 'image', 'cont']:
            losses = [tl[task_name] for tl in task_losses_all]
            task_baselines[task_name] = np.mean(losses)

        fidelity_scores = []
        for tl in task_losses_all:
            score = 0.0
            for task_name in ['reward', 'image', 'cont']:
                relative_error = tl[task_name] / (task_baselines[task_name] + epsilon)
                score += self.task_weights[task_name] * relative_error
            fidelity_scores.append(score)

        scores = np.array(fidelity_scores)
        inv_scores = -scores / sigma
        exp_scores = np.exp(inv_scores - inv_scores.max())
        weights = exp_scores / exp_scores.sum()
        return weights.tolist()

    def _log_fedwm_debug(self, dynamics_weights, encoder_weights, decoder_weights):
        pass

    def _normalize_state_dict_key(self, key: str) -> str:
        if key.startswith('_orig_mod.'):
            return key[len('_orig_mod.'):]
        return key

    def _compute_control_norm(self, control: Dict) -> float:
        norm = 0.0
        for name, c in control['wm'].items():
            norm += torch.sum(c ** 2).item()
        for name, c in control['task_behavior'].items():
            norm += torch.sum(c ** 2).item()
        return np.sqrt(norm)

    def _compute_entropy(self, weights: np.ndarray) -> float:
        return -np.sum(weights * np.log(weights + 1e-8))

    def _deep_copy_weights(self, weights: Dict) -> Dict:
        copy_weights = {'wm': {}, 'task_behavior': {}}
        for key in weights['wm'].keys():
            copy_weights['wm'][key] = weights['wm'][key].clone().detach()
        for key in weights['task_behavior'].keys():
            copy_weights['task_behavior'][key] = weights['task_behavior'][key].clone().detach()
        return copy_weights

    def send_models(self):
        if self.fed_aggregation == 'local':
            return

        for client in self.selected_clients:
            client.agent.set_weights(self.global_weights)

    def save_global_model(self, round_num):
        if self.fed_aggregation == 'local':
            return

        save_path = self.server_logdir / f"global_model_round_{round_num}.pt"
        torch.save(self.global_weights, save_path)
        print(f"Global model saved to {save_path}")

    def evaluate_clients_parallel(self, max_workers=None):
        print("\nEvaluating all clients...")
        start = time.time()

        rewards = {}
        r_sum = 0
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_client = {
                executor.submit(client.evaluate): client
                for client in self.clients
            }

            for future in as_completed(future_to_client):
                client = future_to_client[future]
                try:
                    reward = future.result()
                    rewards[f'client_{client.idx}'] = reward
                    r_sum += reward
                except Exception as e:
                    rewards[f'client_{client.idx}'] = np.nan
                    print(f"Client {client.idx} eval error: {e}")

        valid_rewards = [r for r in rewards.values() if not np.isnan(r)]
        avg_reward = r_sum / len(valid_rewards) if valid_rewards else 0

        print(f'Evaluation rewards: {rewards}')
        print(f'Average reward: {avg_reward:.2f}')
        print(f'Evaluation time: {time.time() - start:.2f}s\n')

        return rewards, avg_reward

    def evaluate_clients(self):
        print("\nEvaluating all clients...")
        start = time.time()

        rewards = {}
        r_sum = 0

        for client in self.selected_clients:
            try:
                reward = client.evaluate()
                rewards[f'client_{client.idx}'] = reward
                r_sum += reward
            except Exception as e:
                rewards[f'client_{client.idx}'] = np.nan
                print(f"Client {client.idx} eval error: {e}")

        valid_rewards = [r for r in rewards.values() if not np.isnan(r)]
        avg_reward = r_sum / len(valid_rewards) if valid_rewards else 0

        print(f'Evaluation rewards: {rewards}')
        print(f'Average reward: {avg_reward:.2f}')
        print(f'Evaluation time: {time.time() - start:.2f}s\n')

        return rewards, avg_reward

    def train(self):
        for round_num in range(self.start_round, self.global_rounds):
            print(f"\n{'#' * 60}")
            print(f"Federated Learning Round {round_num + 1}/{self.global_rounds}")
            print(f"{'#' * 60}\n")

            round_start = time.time()
            self.logger.step = round_num + 1

            self.select_clients()

            for client in self.selected_clients:
                client.train()
                torch.cuda.empty_cache()

            self.receive_models()
            self.aggregate_models()
            self.send_models()

            self.save_checkpoint(round_num + 1)

            if (round_num + 1) % self.eval_gap == 0:
                rewards, avg_reward = self.evaluate_clients_parallel()
                self.logger.scalar("eval/avg_reward", avg_reward)
                self.logger.write()
                self.save_global_model(round_num + 1)

            round_time = time.time() - round_start
            print(f"\nRound {round_num + 1} completed in {round_time:.2f}s")
            print("=" * 60)

            torch.cuda.empty_cache()

        print("\n" + "=" * 60)
        print("Final Evaluation")
        print("=" * 60)
        final_rewards, final_avg = self.evaluate_clients_parallel()
        self.logger.scalar("eval/final_avg_reward", final_avg)
        self.logger.write()

        for client in self.clients:
            client.close_envs()

        print("\n" + "=" * 60)
        print("Federated Learning Training Completed!")
        print("=" * 60)


def main(config):
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()

    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat
    config.time_limit //= config.action_repeat

    logdir = pathlib.Path(config.logdir).expanduser()
    logdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Unified Federated Learning Configuration")
    print(f"{'=' * 60}")
    print(f"Clients: {config.num_clients}")
    print(f"Selection ratio: {config.select_ratio:.1%}")
    print(f"Total rounds: {config.global_rounds}")
    print(f"Local steps per round: {config.local_steps}")
    print(f"Aggregation method: {config.fed_aggregation.upper()}")
    print(f"Evaluation gap: {config.eval_gap}")
    print(f"Task: {config.task}")
    print(f"Seed: {config.seed}")
    print(f"Device: {config.device}")
    print(f"{'=' * 60}\n")

    server = FederatedServer(config)
    server.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+")
    args, remaining = parser.parse_known_args()
    configs = yaml.safe_load(
        (pathlib.Path(sys.argv[0]).parent / "configs.yaml").read_text()
    )


    def recursive_update(base, update):
        for key, value in update.items():
            if isinstance(value, dict) and key in base:
                recursive_update(base[key], value)
            else:
                base[key] = value


    name_list = ["defaults", *args.configs] if args.configs else ["defaults"]
    defaults = {}
    for name in name_list:
        recursive_update(defaults, configs[name])

    parser = argparse.ArgumentParser()
    for key, value in sorted(defaults.items(), key=lambda x: x[0]):
        arg_type = tools.args_type(value)
        parser.add_argument(f"--{key}", type=arg_type, default=arg_type(value))
    main(parser.parse_args(remaining))
