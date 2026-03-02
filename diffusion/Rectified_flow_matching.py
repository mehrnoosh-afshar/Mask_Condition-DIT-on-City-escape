from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PredSpace = Literal["v", "x"]
SamplerType = Literal["euler", "heun"]


@dataclass
class FlowMatchingConfig:
    """
    Rectified Flow / Flow Matching for the linear path:
        z_t = t * x + (1 - t) * eps,   eps ~ N(0, I),  t ~ U(0,1)

    Velocity target:
        v* = d z_t / dt = x - eps

    If model predicts x, conversions:
        v = (x - z_t) / (1 - t)
        x = (1 - t) * v + z_t
    """
    pred: PredSpace = "v"
    loss_space: PredSpace = "v"
    t_sampling: Literal["uniform", "logit_normal"] = "uniform"
    t_logit_normal_mu: float = -0.8
    t_logit_normal_sigma: float = 1.0
    eps_clip_min: float = 1e-4                  # only used for x<->v conversions (not needed for pred="v")
    mse_reduction: Literal["mean", "sum"] = "mean"

    # Sampling
    sampler: SamplerType = "heun"               # "euler" or "heun"
    sample_steps: int = 200
    t0_eps: float = 1e-4                        # avoid exactly t=0 if you want
    use_midpoint_t: bool = True                 # recommended for Euler; OK for Heun too

    # Debug (optional)
    debug_cond_effect_every: int = 0            # 0 disables; otherwise prints every N calls to loss()


class FlowMatching:
    """
    Helper (not nn.Module): provides sampling and flow-matching loss.
    Assumes model signature:
        model(z_t, t, cond, **model_kwargs) -> v_pred  (if cfg.pred="v")
    """

    def __init__(self, cfg: FlowMatchingConfig, device: torch.device):
        self.cfg = cfg
        self.device = device
        self._loss_calls = 0

    # -----------------------------
    # t sampling
    # -----------------------------
    def sample_t(self, batch: int) -> torch.Tensor:
        if self.cfg.t_sampling == "uniform":
            t = torch.rand((batch,), device=self.device)
        elif self.cfg.t_sampling == "logit_normal":
            t = torch.randn((batch,), device=self.device) * self.cfg.t_logit_normal_sigma + self.cfg.t_logit_normal_mu
            t = torch.sigmoid(t)
        else:
            raise ValueError(f"Unknown t_sampling={self.cfg.t_sampling}")

        # Optional clamp away from endpoints for numerical hygiene
        if self.cfg.t0_eps > 0:
            t = t.clamp(self.cfg.t0_eps, 1.0 - self.cfg.t0_eps)
        return t

    # -----------------------------
    # path: z_t
    # -----------------------------
    @staticmethod
    def make_zt(x: torch.Tensor, eps: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # z_t = t*x + (1-t)*eps
        t4 = t.view(-1, 1, 1, 1).to(x.device)
        return t4 * x + (1.0 - t4) * eps.to(x.device)

    # -----------------------------
    # conversions between x and v
    # -----------------------------
    def v_from_x(self, x_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        one_minus_t = (1.0 - t).view(-1, 1, 1, 1).clamp(min=self.cfg.eps_clip_min).to(x_pred.device)
        return (x_pred - z_t) / one_minus_t

    def x_from_v(self, v_pred: torch.Tensor, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        one_minus_t = (1.0 - t).view(-1, 1, 1, 1).to(v_pred.device)
        return one_minus_t * v_pred + z_t

    # -----------------------------
    # loss
    # -----------------------------
    def loss(
        self,
        model: nn.Module,
        x: torch.Tensor,
        cond: torch.Tensor,
        t: Optional[torch.Tensor] = None,
        eps: Optional[torch.Tensor] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Flow-matching MSE in cfg.loss_space ("v" or "x").
        """
        self._loss_calls += 1
        model_kwargs = model_kwargs or {}
        B = x.shape[0]

        if t is None:
            t = self.sample_t(B).to(x.device)
        else:
            t = t.to(x.device)
            if self.cfg.t0_eps > 0:
                t = t.clamp(self.cfg.t0_eps, 1.0 - self.cfg.t0_eps)

        if eps is None:
            eps = torch.randn_like(x)

        z_t = self.make_zt(x, eps, t)

        out = model(z_t, t, cond, **model_kwargs)

        # targets
        v_star = x - eps
        x_star = x

        target = v_star if self.cfg.loss_space == "v" else x_star

        # map prediction into loss space
        if self.cfg.pred == self.cfg.loss_space:
            pred = out
        elif self.cfg.pred == "x" and self.cfg.loss_space == "v":
            pred = self.v_from_x(out, z_t, t)
        elif self.cfg.pred == "v" and self.cfg.loss_space == "x":
            pred = self.x_from_v(out, z_t, t)
        else:
            raise ValueError(f"Unsupported pred/loss combo pred={self.cfg.pred}, loss={self.cfg.loss_space}")

        loss = F.mse_loss(pred, target, reduction=self.cfg.mse_reduction)
        
        # Debug: print loss components on early steps
        if self._loss_calls < 10:
            print(f"[FM debug step {self._loss_calls}] pred stats: mean={pred.mean().item():.4f} std={pred.std().item():.4f} range=[{pred.min().item():.4f}, {pred.max().item():.4f}]")
            print(f"[FM debug step {self._loss_calls}] target stats: mean={target.mean().item():.4f} std={target.std().item():.4f} range=[{target.min().item():.4f}, {target.max().item():.4f}]")
            print(f"[FM debug step {self._loss_calls}] loss: {loss.item():.6f}")

        # Optional debug: conditioning effect (expensive: extra forward)
        if self.cfg.debug_cond_effect_every and (self._loss_calls % self.cfg.debug_cond_effect_every == 0):
            with torch.no_grad():
                perm = torch.randperm(B, device=cond.device)
                out_perm = model(z_t, t, cond[perm], **model_kwargs)
                cond_effect = (out - out_perm).abs().mean()
                baseline = target.pow(2).mean()
                ratio = loss / (baseline + 1e-8)
                print(f"[FM debug] cond_effect={float(cond_effect):.6f} loss={loss.item():.4f} "
                      f"baseline={baseline.item():.4f} ratio={ratio.item():.4f}")

        return loss

    # -----------------------------
    # samplers
    # -----------------------------
    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        cond: torch.Tensor,
        steps: Optional[int] = None,
        device: Optional[torch.device] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        sampler: Optional[SamplerType] = None,
    ) -> torch.Tensor:
        """
        Integrate dz/dt = v_theta(z,t,cond) from t=0 -> 1.
        Choose sampler via cfg.sampler or override arg `sampler`.
        """
        model_kwargs = model_kwargs or {}
        sampler = sampler or self.cfg.sampler
        steps = int(steps or self.cfg.sample_steps)

        device = device or self.device
        B = cond.shape[0]
        cond = cond.to(device)

        # IMPORTANT: do NOT scale z0. For this path, z(t=0)=eps ~ N(0,I).
        z = torch.randn(shape, device=device)

        if sampler == "euler":
            return self._sample_euler(model, z, cond, steps, device, model_kwargs)
        elif sampler == "heun":
            return self._sample_heun(model, z, cond, steps, device, model_kwargs)
        else:
            raise ValueError(f"Unknown sampler={sampler}")

    def _t_at(self, i: int, steps: int, B: int, device: torch.device, *, midpoint: bool) -> torch.Tensor:
        if midpoint:
            tval = (i + 0.5) / float(steps)
        else:
            tval = i / float(steps)
        if self.cfg.t0_eps > 0:
            tval = max(self.cfg.t0_eps, min(1.0 - self.cfg.t0_eps, tval))
        return torch.full((B,), float(tval), device=device)

    @torch.no_grad()
    def _sample_euler(
        self,
        model: nn.Module,
        z: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        device: torch.device,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        dt = 1.0 / float(steps)
        B = cond.shape[0]
        use_mid = bool(self.cfg.use_midpoint_t)

        for i in range(steps):
            t = self._t_at(i, steps, B, device, midpoint=use_mid)

            out = model(z, t, cond, **model_kwargs)

            if self.cfg.pred == "v":
                v = out
            else:
                # model predicts x, convert to v
                one_minus_t = (1.0 - t).view(B, 1, 1, 1).clamp(min=self.cfg.eps_clip_min)
                v = (out - z) / one_minus_t

            z = z + dt * v

        return z

    @torch.no_grad()
    def _sample_heun(
        self,
        model: nn.Module,
        z: torch.Tensor,
        cond: torch.Tensor,
        steps: int,
        device: torch.device,
        model_kwargs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Heun / RK2:
          z_{n+1} = z_n + dt/2 * (v(z_n,t_n) + v(z_n + dt*v0, t_{n+1}))
        """
        dt = 1.0 / float(steps)
        B = cond.shape[0]

        for i in range(steps):
            t0 = self._t_at(i, steps, B, device, midpoint=False)
            t1 = self._t_at(i + 1, steps, B, device, midpoint=False)

            out0 = model(z, t0, cond, **model_kwargs)
            if self.cfg.pred == "v":
                v0 = out0
            else:
                one_minus_t0 = (1.0 - t0).view(B, 1, 1, 1).clamp(min=self.cfg.eps_clip_min)
                v0 = (out0 - z) / one_minus_t0

            z_euler = z + dt * v0

            out1 = model(z_euler, t1, cond, **model_kwargs)
            if self.cfg.pred == "v":
                v1 = out1
            else:
                one_minus_t1 = (1.0 - t1).view(B, 1, 1, 1).clamp(min=self.cfg.eps_clip_min)
                v1 = (out1 - z_euler) / one_minus_t1

            z = z + 0.5 * dt * (v0 + v1)

        return z