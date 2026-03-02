import math
from dataclasses import dataclass
from typing import Any, Optional , Literal

import torch
import torch.nn as nn


# class EMA:
#     """Exponential moving average of parameters."""
#     def __init__(self, model: nn.Module, decay: float = 0.9999):
#         self.decay = decay
#         # Includes buffers too (e.g., running stats). That's usually what you want for EMA eval.
#         self.shadow = {k: v.detach().clone() for k, v in model.state_dict().items()}

#     @torch.no_grad()
#     def update(self, model: nn.Module):
#         msd = model.state_dict()
#         for k, v in msd.items():
#             self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

#     @torch.no_grad()
#     def copy_to(self, model: nn.Module):
#         model.load_state_dict(self.shadow, strict=True)


def make_beta_schedule(
    T: int,
    schedule: str = "linear",
    beta_start: float = 1e-4,
    beta_end: float = 2e-2,
) -> torch.Tensor:
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)

    if schedule == "cosine":
        # cosine schedule (Nichol & Dhariwal style)
        s = 0.008
        steps = T + 1
        x = torch.linspace(0, T, steps)
        alphas_cumprod = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(1e-8, 0.999)

    raise ValueError(f"Unknown schedule: {schedule}")


@dataclass
class DiffusionConfig:
    T: int = 1000
    schedule: str = "linear"   # "linear" or "cosine"
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"     # only "eps" supported here
    sample: str = "ddim"      # "ddpm" or "ddim"
    t_sampling: Literal["uniform", "logit_normal"] = "uniform"


class GaussianDiffusion:
    """
    Classic DDPM training objective (predict eps).
    Includes DDPM and DDIM sampling.
    """
    def __init__(
        self,
        cfg: DiffusionConfig,
        device: torch.device,
        clip_denoised: bool = True,
        # DDIM options
        ddim_steps: int = 50,
        eta: float = 0.0,
    ):
        self.cfg = cfg
        self.device = device
        self.clip_denoised = clip_denoised

        self.ddim_steps = int(ddim_steps)
        self.eta = float(eta)

        betas = make_beta_schedule(cfg.T, cfg.schedule, cfg.beta_start, cfg.beta_end).to(device)
        self.betas = betas                                      # [T]
        self.alphas = 1.0 - betas                               # [T]
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0) # [T]
        self.alphas_cumprod_prev = torch.cat(
            [torch.ones(1, device=device), self.alphas_cumprod[:-1]], dim=0
        )  # [T]

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)                 # [T]
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod) # [T]

        # ----- DDPM posterior q(x_{t-1} | x_t, x0) -----
        # posterior_var = beta_t * (1 - abar_{t-1}) / (1 - abar_t)
        self.posterior_var = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_var = self.posterior_var.clamp(min=1e-20)

        # posterior_mean = coef1 * x0 + coef2 * x_t
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

        # ----- DDIM timestep subset -----
        # Uniform stride; includes 0 and T-1 approximately.
        if self.ddim_steps <= 0:
            raise ValueError("ddim_steps must be >= 1")
        if self.ddim_steps > cfg.T:
            self.ddim_steps = cfg.T

        # e.g., T=1000, steps=50 => [0, 20, 40, ..., 980] (and we reverse it)
        stride = cfg.T // self.ddim_steps
        t_seq = torch.arange(0, cfg.T, stride, device=device, dtype=torch.long)
        # Ensure last timestep is included (T-1)
        if t_seq[-1].item() != cfg.T - 1:
            t_seq = torch.cat([t_seq, torch.tensor([cfg.T - 1], device=device, dtype=torch.long)], dim=0)
        self.ddim_timesteps = t_seq  # ascending; we’ll flip during sampling
    
    # -----------------------------
    # t sampling discrete timesteps for training
    # -----------------------------
    def sample_t(self, batch: int, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Returns t as Long indices in [0, T-1].
        """
        device = device or self.device

        if self.cfg.t_sampling == "uniform":
            return torch.randint(0, self.cfg.T, (batch,), device=device, dtype=torch.long)

        if self.cfg.t_sampling == "logit_normal":
            # sample continuous in (0,1), then map to discrete indices
            u = torch.randn((batch,), device=device) * self.cfg.t_logit_normal_sigma + self.cfg.t_logit_normal_mu
            t_cont = torch.sigmoid(u)  # (0,1)
            t_idx = torch.clamp((t_cont * (self.cfg.T - 1)).round(), 0, self.cfg.T - 1).long()
            return t_idx
    

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape):
        """
        a: [T], t: [B] (long), returns [B,1,1,1,...] broadcastable to x
        """
        if t.dtype != torch.long:
            t = t.long()
        t = t.to(a.device)
        B = t.shape[0]
        out = a.gather(0, t).view(B, *([1] * (len(x_shape) - 1)))
        return out

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sa = self._extract(self.sqrt_alphas_cumprod, t, x0.shape)
        so = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sa * x0 + so * noise

    def loss(self, model: nn.Module, x: torch.Tensor, cond: torch.Tensor, t: Optional[torch.Tensor]=None,) -> torch.Tensor:
        if self.cfg.objective != "eps":
            raise ValueError(f"Only objective='eps' is supported, got {self.cfg.objective}")
        noise = torch.randn_like(x)

        B = x.shape[0]
        device = x.device

        if t is None:
            t = self.sample_t(B)

        xt = self.q_sample(x, t, noise=noise)
        pred = model(xt, t.long(), cond)  # predicted eps
        return torch.mean((pred - noise) ** 2)

    @torch.no_grad()
    def p_sample_step(self, model: nn.Module, x_t: torch.Tensor, t: torch.Tensor, cond: Any = None) -> torch.Tensor:
        """
        Sample x_{t-1} from x_t using DDPM ancestral sampling.
        t: [B] (long)
        """
        t = t.long()
        eps = model(x_t, t, cond)

        abar_t = self._extract(self.alphas_cumprod, t, x_t.shape)
        sqrt_abar_t = torch.sqrt(abar_t)
        sqrt_one_minus_abar_t = torch.sqrt(1.0 - abar_t)

        x0 = (x_t - sqrt_one_minus_abar_t * eps) / (sqrt_abar_t + 1e-8)
        if self.clip_denoised:
            x0 = x0.clamp(-1.0, 1.0)

        c1 = self._extract(self.posterior_mean_coef1, t, x_t.shape)
        c2 = self._extract(self.posterior_mean_coef2, t, x_t.shape)
        mean = c1 * x0 + c2 * x_t

        var = self._extract(self.posterior_var, t, x_t.shape)
        noise = torch.randn_like(x_t)

        nonzero_mask = (t != 0).float().view(x_t.shape[0], *([1] * (x_t.ndim - 1)))
        return mean + nonzero_mask * torch.sqrt(var) * noise

    @torch.no_grad()
    def sample_ddpm(
        self,
        model: nn.Module,
        shape,
        cond: Any = None,
        device: Optional[torch.device] = None,
        x_T: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        device = device or next(model.parameters()).device
        x = x_T if x_T is not None else torch.randn(shape, device=device)

        for ti in reversed(range(self.cfg.T)):
            t = torch.full((shape[0],), ti, device=device, dtype=torch.long)
            x = self.p_sample_step(model, x, t, cond=cond)

        return x

    @torch.no_grad()
    def sample_ddim(
        self,
        model: nn.Module,
        shape,
        cond: Any = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """
        Deterministic DDIM when eta=0. Stochastic when eta>0.
        """
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device)

        t_seq = self.ddim_timesteps.to(device)
        t_seq = torch.flip(t_seq, dims=[0])  # descending (high -> low)

        for i in range(len(t_seq)):
            t = t_seq[i].repeat(shape[0])  # [B]
            eps = model(x, t, cond)

            alpha_bar_t = self._extract(self.alphas_cumprod, t, x.shape)
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

            x0 = (x - sqrt_one_minus_alpha_bar_t * eps) / (sqrt_alpha_bar_t + 1e-8)
            if self.clip_denoised:
                x0 = x0.clamp(-1.0, 1.0)

            if i == len(t_seq) - 1:
                x = x0
                break

            t_prev = t_seq[i + 1].repeat(shape[0])
            alpha_bar_prev = self._extract(self.alphas_cumprod, t_prev, x.shape)

            sigma = self.eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t + 1e-8)) * \
                    torch.sqrt(1.0 - alpha_bar_t / (alpha_bar_prev + 1e-8))

            noise = torch.randn_like(x) if self.eta > 0 else torch.zeros_like(x)
            dir_xt = torch.sqrt(torch.clamp(1.0 - alpha_bar_prev - sigma**2, min=0.0)) * eps

            x = torch.sqrt(alpha_bar_prev) * x0 + dir_xt + sigma * noise

        return x
    
    def sample(self, model: nn.Module, shape, cond: Any = None, device: Optional[torch.device] = None) -> torch.Tensor:
        if self.cfg.sample == "ddpm":
            return self.sample_ddpm(model, shape, cond, device)
        elif self.cfg.sample == "ddim":
            return self.sample_ddim(model, shape, cond, device)
        else:
            raise ValueError(f"Unknown sample method: {self.cfg.sample}")
