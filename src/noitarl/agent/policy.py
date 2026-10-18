from __future__ import annotations

import copy
from collections import deque
from collections.abc import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ot.partial import entropic_partial_wasserstein
from torch.nn.utils import clip_grad_norm_
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from noitarl.agent.encoder import FanInInitReLULayer, ImpalaEncoder
from noitarl.environment.crafter_score import ACHIEVEMENTS
from noitarl.settings import AgentSettings, DistillSettings


class NormalizeEwma(nn.Module):
    def __init__(self, insize: int, beta: float = 0.99, epsilon: float = 1e-2) -> None:
        super().__init__()
        self.beta = beta
        self.epsilon = epsilon
        self.running_mean = nn.Parameter(torch.zeros(insize), requires_grad=False)
        self.running_mean_sq = nn.Parameter(torch.zeros(insize), requires_grad=False)
        self.debiasing_term = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def running_mean_var(self) -> tuple[torch.Tensor, torch.Tensor]:
        debiased = self.debiasing_term.clamp(min=self.epsilon)
        mean = self.running_mean / debiased
        mean_sq = self.running_mean_sq / debiased
        var = (mean_sq - mean**2).clamp(min=1e-2)
        return mean, var

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_mean = x.detach().mean(dim=0)
            batch_mean_sq = (x.detach() ** 2).mean(dim=0)
            weight = self.beta
            self.running_mean.mul_(weight).add_(batch_mean * (1.0 - weight))
            self.running_mean_sq.mul_(weight).add_(batch_mean_sq * (1.0 - weight))
            self.debiasing_term.mul_(weight).add_(1.0 * (1.0 - weight))
        mean, var = self.running_mean_var()
        return (x - mean) / torch.sqrt(var)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        mean, var = self.running_mean_var()
        return x * torch.sqrt(var) + mean


class CategoricalActionHead(nn.Module):
    def __init__(self, insize: int, num_actions: int, init_scale: float = 0.01) -> None:
        super().__init__()
        self.linear = nn.Linear(insize, num_actions)
        nn.init.orthogonal_(self.linear.weight, gain=init_scale)
        nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.linear(x), dim=-1)

    def log_prob(self, logits: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return torch.gather(logits, dim=-1, index=actions)

    def entropy(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.exp(logits)
        return -torch.sum(probs * logits, dim=-1, keepdim=True)

    def sample(self, logits: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return torch.argmax(logits, dim=-1, keepdim=True)
        uniform = torch.rand_like(logits)
        uniform[uniform == 1.0] = 0.999
        gumbels = logits - torch.log(-torch.log(uniform))
        return torch.argmax(gumbels, dim=-1, keepdim=True)

    def kl_divergence(self, logits_q: torch.Tensor, logits_p: torch.Tensor) -> torch.Tensor:
        return torch.sum(torch.exp(logits_q) * (logits_q - logits_p), dim=-1, keepdim=True)


class ScaledMSEHead(nn.Module):
    def __init__(self, insize: int, ewma: float = 0.99) -> None:
        super().__init__()
        self.linear = nn.Linear(insize, 1)
        nn.init.orthogonal_(self.linear.weight, gain=0.1)
        nn.init.constant_(self.linear.bias, 0.0)
        self.normalizer = NormalizeEwma(1, beta=ewma)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalizer.denormalize(x)

    def mse_loss(self, pred: torch.Tensor, targ: torch.Tensor) -> torch.Tensor:
        return F.mse_loss(pred, self.normalizer(targ), reduction="none")


class MLP(nn.Module):
    def __init__(self, insize: int, nhidlayer: int, outsize: int, hidsize: int) -> None:
        super().__init__()
        sizes = [insize] + nhidlayer * [hidsize] + [outsize]
        layers: list[nn.Module] = []
        for index in range(len(sizes) - 1):
            active = index < nhidlayer
            layers.append(
                FanInInitReLULayer(
                    sizes[index],
                    sizes[index + 1],
                    layer_type="linear",
                    use_activation=active,
                    init_scale=1.4 if active else 1.0,
                    layer_norm=active,
                )
            )
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class CrafterPolicy(nn.Module):
    def __init__(self, num_actions: int, settings: AgentSettings) -> None:
        super().__init__()
        self._settings = settings
        self._distill = settings.distill
        hidden = settings.encoder.hidden
        encoder_out = settings.encoder.outsize
        self.enc = ImpalaEncoder(
            (3, 64, 64),
            settings.encoder.channels,
            encoder_out,
            settings.encoder.nblock,
        )
        self.linear = FanInInitReLULayer(
            encoder_out, hidden, layer_type="linear", init_scale=1.4, layer_norm=True
        )
        self.use_memory = settings.distill.enabled
        head_in = 2 * hidden if self.use_memory else hidden
        self.pi_head = CategoricalActionHead(head_in, num_actions)
        self.vf_head = ScaledMSEHead(head_in, ewma=settings.value_ewma)
        self.action_mlp = MLP(num_actions, 1, 2 * hidden, hidden)
        self.next_goal_pred_mlp = MLP(2 * hidden, 1, hidden, hidden)
        self._num_actions = num_actions
        self._hidden = hidden

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.enc(obs))

    def forward(self, obs: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
        latents = self.encode(obs)
        if self.use_memory:
            merged = torch.cat([latents, states], dim=-1)
        else:
            merged = latents
        return {
            "latents": latents,
            "pi_logits": self.pi_head(merged),
            "vpreds": self.vf_head(merged),
        }

    @torch.no_grad()
    def act(self, obs: torch.Tensor, states: torch.Tensor) -> dict[str, torch.Tensor]:
        was_training = self.training
        self.eval()
        outputs = self.forward(obs, states)
        pi_logits = outputs["pi_logits"]
        actions = self.pi_head.sample(pi_logits)
        log_probs = self.pi_head.log_prob(pi_logits, actions)
        vpreds = self.vf_head.denormalize(outputs["vpreds"])
        outputs.update({"actions": actions, "log_probs": log_probs, "vpreds": vpreds})
        if was_training:
            self.train()
        return outputs

    def compute_losses(
        self,
        obs: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        log_probs: torch.Tensor,
        vtargs: torch.Tensor,
        advs: torch.Tensor,
        clip_param: float,
    ) -> dict[str, torch.Tensor]:
        outputs = self.forward(obs, states)
        pi_logits = outputs["pi_logits"]
        new_log_probs = self.pi_head.log_prob(pi_logits, actions)
        ratio = torch.exp(new_log_probs - log_probs)
        clipped = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
        pi_loss = -torch.min(advs * ratio, advs * clipped).mean()
        vf_loss = self.vf_head.mse_loss(outputs["vpreds"], vtargs).mean()
        entropy = self.pi_head.entropy(pi_logits).mean()
        return {"pi_loss": pi_loss, "vf_loss": vf_loss, "entropy": entropy}

    def film(self, latents: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        onehot = torch.eye(self._num_actions, device=actions.device)[actions.squeeze(-1)]
        action_latents = self.action_mlp(onehot)
        gamma, beta = torch.chunk(action_latents, 2, dim=-1)
        return (1 + gamma) * latents + beta

    def get_states(self, goal_obs: torch.Tensor, goal_next_obs: torch.Tensor) -> torch.Tensor:
        zero_obs = goal_obs.reshape(goal_obs.size(0), -1).eq(0).all(dim=-1, keepdim=True)
        zero_next = goal_next_obs.reshape(goal_next_obs.size(0), -1).eq(0).all(dim=-1, keepdim=True)
        goal_latents = self.encode(goal_obs)
        goal_next_latents = self.encode(goal_next_obs)
        goal_latents = torch.where(zero_obs, torch.zeros_like(goal_latents), goal_latents)
        goal_next_latents = torch.where(
            zero_next, torch.zeros_like(goal_next_latents), goal_next_latents
        )
        states = goal_next_latents - goal_latents
        return F.normalize(states, dim=-1)

    def get_next_goal_preds(
        self, latents: torch.Tensor, actions: torch.Tensor, states: torch.Tensor
    ) -> torch.Tensor:
        latents = self.film(latents, actions)
        if self.use_memory:
            latents = torch.cat([latents, states], dim=-1)
        preds = self.next_goal_pred_mlp(latents)
        return F.normalize(preds, dim=-1)

    def compute_pred_losses(
        self,
        anc_goal_obs: torch.Tensor,
        anc_goal_next_obs: torch.Tensor,
        pos_obs: torch.Tensor,
        pos_actions: torch.Tensor,
        pos_old_states: torch.Tensor,
        pos_old_vtargs: torch.Tensor,
        neg_obs: torch.Tensor,
        neg_actions: torch.Tensor,
        neg_old_states: torch.Tensor,
        neg_old_vtargs: torch.Tensor,
        old_model: CrafterPolicy,
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)
        pos_outputs = self.forward(pos_obs, pos_old_states)
        pos_old_outputs = old_model.act(pos_obs, pos_old_states)
        pos_preds = self.get_next_goal_preds(
            pos_outputs["latents"], pos_actions, pos_old_states
        )
        neg_outputs = self.forward(neg_obs, neg_old_states)
        neg_old_outputs = old_model.act(neg_obs, neg_old_states)
        neg_preds = self.get_next_goal_preds(
            neg_outputs["latents"], neg_actions, neg_old_states
        )
        temperature = self._distill.temperature
        pos_logits = torch.einsum("bk,bk->b", anc_states, pos_preds)
        neg_logits = torch.einsum("bk,bk->b", anc_states, neg_preds)
        logits = torch.stack([pos_logits, neg_logits], dim=-1) / temperature
        targets = torch.zeros(len(logits), device=logits.device, dtype=torch.long)
        pred_loss = F.cross_entropy(logits, targets)
        pi_logits = torch.cat([pos_outputs["pi_logits"], neg_outputs["pi_logits"]], dim=0)
        old_pi_logits = torch.cat(
            [pos_old_outputs["pi_logits"], neg_old_outputs["pi_logits"]], dim=0
        )
        pi_dist = self.pi_head.kl_divergence(pi_logits, old_pi_logits).mean()
        vpreds = torch.cat([pos_outputs["vpreds"], neg_outputs["vpreds"]], dim=0)
        old_vtargs = torch.cat([pos_old_vtargs, neg_old_vtargs], dim=0)
        vf_dist = self.vf_head.mse_loss(vpreds, old_vtargs).mean()
        return {"pred_loss": pred_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}

    def compute_match_losses(
        self,
        anc_goal_obs: torch.Tensor,
        anc_goal_next_obs: torch.Tensor,
        pos_goal_obs: torch.Tensor,
        pos_goal_next_obs: torch.Tensor,
        neg_goal_obs: torch.Tensor,
        neg_goal_next_obs: torch.Tensor,
        obs: torch.Tensor,
        old_states: torch.Tensor,
        old_vtargs: torch.Tensor,
        old_model: CrafterPolicy,
    ) -> dict[str, torch.Tensor]:
        anc_states = self.get_states(anc_goal_obs, anc_goal_next_obs)
        with torch.no_grad():
            pos_states = self.get_states(pos_goal_obs, pos_goal_next_obs)
            neg_states = self.get_states(neg_goal_obs, neg_goal_next_obs)
        outputs = self.forward(obs, old_states)
        old_outputs = old_model.act(obs, old_states)
        temperature = self._distill.temperature
        pos_logits = torch.einsum("bk,bk->b", anc_states, pos_states)
        neg_logits = torch.einsum("bk,bk->b", anc_states, neg_states)
        logits = torch.stack([pos_logits, neg_logits], dim=-1) / temperature
        targets = torch.zeros(len(logits), device=logits.device, dtype=torch.long)
        match_loss = F.cross_entropy(logits, targets)
        pi_dist = self.pi_head.kl_divergence(
            outputs["pi_logits"], old_outputs["pi_logits"]
        ).mean()
        vf_dist = self.vf_head.mse_loss(outputs["vpreds"], old_vtargs).mean()
        return {"match_loss": match_loss, "pi_dist": pi_dist, "vf_dist": vf_dist}


class RolloutStorage:
    def __init__(
        self,
        nstep: int,
        nproc: int,
        obs_shape: tuple[int, ...],
        hidsize: int,
        device: torch.device,
    ) -> None:
        self.nstep = nstep
        self.nproc = nproc
        self.device = device
        self.obs = torch.zeros(nstep + 1, nproc, *obs_shape, device=device)
        self.actions = torch.zeros(nstep, nproc, 1, device=device, dtype=torch.long)
        self.rewards = torch.zeros(nstep, nproc, 1, device=device)
        self.masks = torch.ones(nstep + 1, nproc, 1, device=device)
        self.vpreds = torch.zeros(nstep + 1, nproc, 1, device=device)
        self.log_probs = torch.zeros(nstep, nproc, 1, device=device)
        self.returns = torch.zeros(nstep, nproc, 1, device=device)
        self.advs = torch.zeros(nstep, nproc, 1, device=device)
        self.successes = torch.zeros(
            nstep + 1, nproc, len(ACHIEVEMENTS), device=device, dtype=torch.long
        )
        self.states = torch.zeros(nstep + 1, nproc, hidsize, device=device)
        self.step = 0

    def get_inputs(self, step: int) -> dict[str, torch.Tensor]:
        index = step if step >= 0 else self.nstep
        return {"obs": self.obs[index], "states": self.states[index]}

    def insert(
        self,
        model: CrafterPolicy,
        obs: torch.Tensor,
        latents: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        masks: torch.Tensor,
        vpreds: torch.Tensor,
        log_probs: torch.Tensor,
        successes: torch.Tensor,
    ) -> None:
        prev_successes = self.successes[self.step]
        prev_states = self.states[self.step]
        success_changed = (successes != prev_successes).any(dim=-1, keepdim=True)
        if success_changed.any():
            with torch.no_grad():
                next_latents = model.encode(obs)
                states = next_latents - latents
                states = F.normalize(states, dim=-1)
                states = torch.where(success_changed, states, prev_states)
        else:
            states = prev_states
        done = masks == 0
        successes = torch.where(done, torch.zeros_like(successes), successes)
        states = torch.where(done, torch.zeros_like(states), states)
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        self.vpreds[self.step].copy_(vpreds)
        self.log_probs[self.step].copy_(log_probs)
        self.successes[self.step + 1].copy_(successes)
        self.states[self.step + 1].copy_(states)
        self.step = (self.step + 1) % self.nstep

    def reset(self) -> None:
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])
        self.successes[0].copy_(self.successes[-1])
        self.states[0].copy_(self.states[-1])
        self.step = 0

    def compute_returns(self, gamma: float, gae_lambda: float) -> None:
        gae = 0.0
        for step in reversed(range(self.nstep)):
            delta = (
                self.rewards[step]
                + gamma * self.vpreds[step + 1] * self.masks[step + 1]
                - self.vpreds[step]
            )
            gae = delta + gamma * gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + self.vpreds[step]
            self.advs[step] = gae
        self.advs = (self.advs - self.advs.mean()) / (self.advs.std() + 1e-8)

    def batches(self, nbatch: int) -> Iterator[dict[str, torch.Tensor]]:
        ndata = self.nstep * self.nproc
        batch_size = ndata // nbatch
        obs = self.obs[:-1].reshape(ndata, *self.obs.shape[2:])
        states = self.states[:-1].reshape(ndata, self.states.shape[-1])
        actions = self.actions.reshape(ndata, 1)
        vtargs = self.returns.reshape(ndata, 1)
        log_probs = self.log_probs.reshape(ndata, 1)
        advs = self.advs.reshape(ndata, 1)
        sampler = BatchSampler(
            SubsetRandomSampler(range(ndata)), batch_size=batch_size, drop_last=True
        )
        for indices in sampler:
            yield {
                "obs": obs[indices],
                "states": states[indices],
                "actions": actions[indices],
                "vtargs": vtargs[indices],
                "log_probs": log_probs[indices],
                "advs": advs[indices],
            }


class DistillBuffer:
    def __init__(self, maxlen: int) -> None:
        self._segments: deque[dict[str, torch.Tensor]] = deque(maxlen=maxlen)
        self.trajs: list[dict[str, torch.Tensor]] = []

    def insert(self, segment: dict[str, torch.Tensor]) -> None:
        self._segments.append(segment)

    def parse(self) -> None:
        self.trajs.clear()
        obs = torch.cat([segment["obs"][:-1] for segment in self._segments], dim=0)
        actions = torch.cat([segment["actions"] for segment in self._segments], dim=0)
        states = torch.cat([segment["states"][:-1] for segment in self._segments], dim=0)
        returns = torch.cat([segment["returns"] for segment in self._segments], dim=0)
        masks = torch.cat([segment["masks"][:-1] for segment in self._segments], dim=0)
        rewards = torch.cat([segment["rewards"] for segment in self._segments], dim=0)
        successes = torch.cat([segment["successes"][:-1] for segment in self._segments], dim=0)
        nproc = obs.shape[1]
        for proc in range(nproc):
            obs_p = obs[:, proc]
            actions_p = actions[:, proc]
            states_p = states[:, proc]
            returns_p = returns[:, proc]
            rewards_p = rewards[:, proc]
            successes_p = successes[:, proc]
            masks_p = masks[:, proc]
            done_steps = (masks_p == 0).squeeze(-1).nonzero().squeeze(-1).tolist()
            if not done_steps:
                continue
            if not isinstance(done_steps, list):
                done_steps = [done_steps]
            for start, end in zip(done_steps[:-1], done_steps[1:]):
                goals = _goals_from_trajectory(
                    obs_p[start:end], rewards_p[start:end], successes_p[start:end]
                )
                self.trajs.append(
                    {
                        "obs": obs_p[start:end],
                        "actions": actions_p[start:end],
                        "old_states": states_p[start:end],
                        "old_vtargs": returns_p[start:end],
                        **goals,
                    }
                )

    def pred_batches(
        self, device: torch.device, max_batch_size: int = 512
    ) -> Iterator[dict[str, torch.Tensor]]:
        for index in torch.randperm(len(self.trajs)):
            traj = self.trajs[int(index)]
            goal_steps = traj["goal_steps"]
            if len(goal_steps) == 0:
                continue
            next_goal_obs, next_goal_next_obs = _next_goals(
                goal_steps, traj["goal_obs"], traj["goal_next_obs"], traj["obs"]
            )
            obs = traj["obs"]
            ndata = len(obs)
            perm = torch.randperm(ndata)
            sampler = BatchSampler(
                SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False
            )
            for inds in sampler:
                yield {
                    "anc_goal_obs": next_goal_obs[inds].to(device),
                    "anc_goal_next_obs": next_goal_next_obs[inds].to(device),
                    "pos_obs": obs[inds].to(device),
                    "pos_actions": traj["actions"][inds].to(device),
                    "pos_old_states": traj["old_states"][inds].to(device),
                    "pos_old_vtargs": traj["old_vtargs"][inds].to(device),
                    "neg_obs": obs[perm[inds]].to(device),
                    "neg_actions": traj["actions"][perm[inds]].to(device),
                    "neg_old_states": traj["old_states"][perm[inds]].to(device),
                    "neg_old_vtargs": traj["old_vtargs"][perm[inds]].to(device),
                }

    def match_batches(
        self,
        model: CrafterPolicy,
        device: torch.device,
        settings: DistillSettings,
        max_batch_size: int = 512,
    ) -> Iterator[dict[str, torch.Tensor]]:
        trajs = [traj for traj in self.trajs if len(traj["goal_steps"]) > 0]
        for index in torch.randperm(len(trajs)):
            traj_s = trajs[int(index)]
            with torch.no_grad():
                states_s = model.get_states(
                    traj_s["goal_obs"].to(device), traj_s["goal_next_obs"].to(device)
                )
            anc_goal_obs: list[torch.Tensor] = []
            anc_goal_next_obs: list[torch.Tensor] = []
            pos_goal_obs: list[torch.Tensor] = []
            pos_goal_next_obs: list[torch.Tensor] = []
            neg_goal_obs: list[torch.Tensor] = []
            neg_goal_next_obs: list[torch.Tensor] = []
            candidates = torch.randperm(len(trajs) - 1)[:16]
            for raw in candidates:
                j = int(raw)
                if j >= index:
                    j += 1
                traj_t = trajs[j]
                with torch.no_grad():
                    states_t = model.get_states(
                        traj_t["goal_obs"].to(device), traj_t["goal_next_obs"].to(device)
                    )
                cost = 1 - torch.einsum("ik,jk->ij", states_s, states_t).cpu().numpy()
                transport = entropic_partial_wasserstein(
                    np.ones(len(states_s)),
                    np.ones(len(states_t)),
                    cost,
                    reg=settings.transport_reg,
                    numItermax=100,
                )
                matrix = torch.from_numpy(transport).float()
                row_inds, col_inds = torch.where(matrix > settings.match_threshold)
                if len(row_inds) == 0:
                    continue
                anc_goal_obs.append(traj_s["goal_obs"][row_inds])
                anc_goal_next_obs.append(traj_s["goal_next_obs"][row_inds])
                pos_goal_obs.append(traj_t["goal_obs"][col_inds])
                pos_goal_next_obs.append(traj_t["goal_next_obs"][col_inds])
                rand_inds = torch.randint(len(traj_t["goal_obs"]), (len(col_inds),))
                neg_goal_obs.append(traj_t["goal_obs"][rand_inds])
                neg_goal_next_obs.append(traj_t["goal_next_obs"][rand_inds])
            if not anc_goal_obs:
                continue
            anc_goal_obs_t = torch.cat(anc_goal_obs, dim=0)
            anc_goal_next_obs_t = torch.cat(anc_goal_next_obs, dim=0)
            pos_goal_obs_t = torch.cat(pos_goal_obs, dim=0)
            pos_goal_next_obs_t = torch.cat(pos_goal_next_obs, dim=0)
            neg_goal_obs_t = torch.cat(neg_goal_obs, dim=0)
            neg_goal_next_obs_t = torch.cat(neg_goal_next_obs, dim=0)
            ndata = len(anc_goal_obs_t)
            rand_inds = torch.randint(len(traj_s["obs"]), (ndata,))
            obs = traj_s["obs"][rand_inds]
            old_states = traj_s["old_states"][rand_inds]
            old_vtargs = traj_s["old_vtargs"][rand_inds]
            sampler = BatchSampler(
                SubsetRandomSampler(range(ndata)), batch_size=max_batch_size, drop_last=False
            )
            for inds in sampler:
                yield {
                    "anc_goal_obs": anc_goal_obs_t[inds].to(device),
                    "anc_goal_next_obs": anc_goal_next_obs_t[inds].to(device),
                    "pos_goal_obs": pos_goal_obs_t[inds].to(device),
                    "pos_goal_next_obs": pos_goal_next_obs_t[inds].to(device),
                    "neg_goal_obs": neg_goal_obs_t[inds].to(device),
                    "neg_goal_next_obs": neg_goal_next_obs_t[inds].to(device),
                    "obs": obs[inds].to(device),
                    "old_states": old_states[inds].to(device),
                    "old_vtargs": old_vtargs[inds].to(device),
                }


def _goals_from_trajectory(
    obs: torch.Tensor, rewards: torch.Tensor, successes: torch.Tensor
) -> dict[str, torch.Tensor]:
    goal_steps = (rewards[:-1] > 0.1).squeeze(-1).nonzero().squeeze(-1) + 1
    if len(goal_steps) == 0:
        empty = torch.zeros(0, *obs.shape[1:])
        return {"goal_steps": goal_steps, "goal_obs": empty, "goal_next_obs": empty}
    goal_obs = obs[goal_steps - 1]
    goal_next_obs = obs[goal_steps]
    return {"goal_steps": goal_steps, "goal_obs": goal_obs, "goal_next_obs": goal_next_obs}


def _next_goals(
    goal_steps: torch.Tensor,
    goal_obs: torch.Tensor,
    goal_next_obs: torch.Tensor,
    obs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    steps = goal_steps.tolist()
    if not isinstance(steps, list):
        steps = [steps]
    boundaries = sorted([0] + steps + [len(obs)])
    next_goal_obs: list[torch.Tensor] = []
    next_goal_next_obs: list[torch.Tensor] = []
    for index, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        span = end - start
        if index == len(boundaries) - 2:
            next_ob = obs[-1].unsqueeze(0).repeat(span, 1, 1, 1)
            next_next_ob = torch.zeros_like(obs[-1]).unsqueeze(0).repeat(span, 1, 1, 1)
        else:
            next_ob = goal_obs[index].unsqueeze(0).repeat(span, 1, 1, 1)
            next_next_ob = goal_next_obs[index].unsqueeze(0).repeat(span, 1, 1, 1)
        next_goal_obs.append(next_ob)
        next_goal_next_obs.append(next_next_ob)
    return torch.cat(next_goal_obs, dim=0), torch.cat(next_goal_next_obs, dim=0)


def run_ppo_update(
    model: CrafterPolicy,
    storage: RolloutStorage,
    optimizer: torch.optim.Optimizer,
    settings: AgentSettings,
) -> None:
    model.train()
    for _ in range(settings.ppo_epochs):
        for batch in storage.batches(settings.n_minibatches):
            losses = model.compute_losses(**batch, clip_param=settings.clip_range)
            loss = (
                losses["pi_loss"]
                + settings.vf_coef * losses["vf_loss"]
                - settings.ent_coef * losses["entropy"]
            )
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            optimizer.step()


def run_distill_update(
    model: CrafterPolicy,
    buffer: DistillBuffer,
    optimizers: tuple[torch.optim.Optimizer, torch.optim.Optimizer],
    settings: AgentSettings,
    device: torch.device,
) -> None:
    if not settings.distill.enabled:
        return
    buffer.parse()
    if not buffer.trajs:
        return
    old_model = copy.deepcopy(model)
    old_model.eval()
    match_optimizer, pred_optimizer = optimizers
    distill = settings.distill
    for _ in range(distill.aux_epochs):
        for batch in buffer.match_batches(model, device, distill):
            losses = model.compute_match_losses(**batch, old_model=old_model)
            loss = (
                losses["match_loss"]
                + distill.pi_coef * losses["pi_dist"]
                + distill.vf_coef * losses["vf_dist"]
            )
            match_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            match_optimizer.step()
        for batch in buffer.pred_batches(device):
            losses = model.compute_pred_losses(**batch, old_model=old_model)
            loss = (
                losses["pred_loss"]
                + distill.pi_coef * losses["pi_dist"]
                + distill.vf_coef * losses["vf_dist"]
            )
            pred_optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), settings.max_grad_norm)
            pred_optimizer.step()
