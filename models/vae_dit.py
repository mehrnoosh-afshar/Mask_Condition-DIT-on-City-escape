from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.models import AutoencoderKL


# ----------------------------
# Timestep embed + DiT blocks
# ----------------------------
class TimestepEmbedder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim, bias=True),
            nn.SiLU(),
            nn.Linear(dim, dim, bias=True),
        )

        half = dim // 2
        emb = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -emb)
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if not torch.is_floating_point(t):
            t = t.float()
        args = t.unsqueeze(1) * self.freqs.unsqueeze(0)  # [B, half]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, dim]
        return self.mlp(emb)


class DiTBlock(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)

        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(model_dim, 6 * model_dim),
        )

        self.attn = nn.MultiheadAttention(model_dim, n_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(model_dim, mlp_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(mlp_dim, model_dim),
        )

        # AdaLN-Zero init
        nn.init.constant_(self.cond_mlp[-1].weight, 0.0)
        nn.init.constant_(self.cond_mlp[-1].bias, 0.0)

    def forward(
        self,
        x: torch.Tensor,  # [B, N, D]
        cond: torch.Tensor,  # [B, D]
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        params = self.cond_mlp(cond).unsqueeze(1)  # [B,1,6D]
        scale1, shift1, gate1, scale2, shift2, gate2 = torch.chunk(params, 6, dim=-1)

        h = self.norm1(x)
        h = h * (1 + scale1) + shift1
        h = self.attn(h, h, h, need_weights=False, key_padding_mask=key_padding_mask)[0]
        x = x + gate1 * h

        h = self.norm2(x)
        h = h * (1 + scale2) + shift2
        h = self.mlp(h)
        x = x + gate2 * h
        return x


class TinyDiTLatent(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        n_layers: int = 12,
        patch_size: int = 2,
        latent_channels: int = 4,
        cond_channels: int = 19,
        latent_hw: Tuple[int, int] = (32, 32),
        n_heads: int = 8,
        mlp_dim: int = 2048,
        n_adaln_cond_cls: Optional[int] = None,
    ):
        super().__init__()
        self.model_dim = model_dim
        self.patch_size = patch_size
        self.latent_channels = latent_channels
        self.cond_channels = cond_channels
        self.Hl, self.Wl = latent_hw

        if self.Hl % patch_size != 0 or self.Wl % patch_size != 0:
            raise ValueError("latent_hw must be divisible by patch_size")

        self.nph = self.Hl // patch_size
        self.npw = self.Wl // patch_size
        self.n_patches = self.nph * self.npw

        self.patch_conv = nn.Conv2d(
            in_channels=latent_channels + cond_channels,
            out_channels=model_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.pos_emb = nn.Parameter(torch.randn(1, self.n_patches, model_dim) * 0.02)
        self.t_embed = TimestepEmbedder(model_dim)

        self.use_adaln_cond = n_adaln_cond_cls is not None
        if self.use_adaln_cond:
            self.adaln_cond_emb = nn.Embedding(n_adaln_cond_cls, model_dim)

        self.blocks = nn.ModuleList([DiTBlock(model_dim, n_heads, mlp_dim) for _ in range(n_layers)])

        self.final_norm = nn.LayerNorm(model_dim)
        self.final_adaln = nn.Sequential(nn.SiLU(), nn.Linear(model_dim, 2 * model_dim))
        self.out_head = nn.Linear(model_dim, (patch_size * patch_size) * latent_channels)

        nn.init.constant_(self.final_adaln[-1].weight, 0.0)
        nn.init.constant_(self.final_adaln[-1].bias, 0.0)

    def forward(
        self,
        z_noisy: torch.Tensor,          # [B, 4, Hl, Wl] scaled latents
        t: torch.Tensor,                # [B]
        mask_onehot_lat: torch.Tensor,  # [B, 19, Hl, Wl]
        adaln_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        b, c, hl, wl = z_noisy.shape
        if (hl, wl) != (self.Hl, self.Wl):
            raise ValueError(f"Expected latent size {(self.Hl, self.Wl)}, got {(hl, wl)}")
        if mask_onehot_lat.shape[0] != b or mask_onehot_lat.shape[2:] != (hl, wl):
            raise ValueError(f"mask_onehot_lat must be [B, {self.cond_channels}, Hl, Wl] aligned with z")

        x = torch.cat([z_noisy, mask_onehot_lat], dim=1)  # [B, 4+19, Hl, Wl]
        tok = self.patch_conv(x)                          # [B, D, nph, npw]
        tok = tok.permute(0, 2, 3, 1).reshape(b, self.n_patches, self.model_dim)  # [B, N, D]
        tok = tok + self.pos_emb

        cond = self.t_embed(t)  # [B, D]
        if self.use_adaln_cond and adaln_cond is not None:
            cond = cond + self.adaln_cond_emb(adaln_cond)

        for blk in self.blocks:
            tok = blk(tok, cond)

        tok = self.final_norm(tok)
        scale_shift = self.final_adaln(cond).unsqueeze(1)  # [B,1,2D]
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)
        tok = tok * (1 + scale) + shift

        out = self.out_head(tok)  # [B, N, p*p*4]
        out = out.reshape(
            b, self.nph, self.npw, self.latent_channels, self.patch_size, self.patch_size
        ).permute(0, 3, 1, 4, 2, 5)
        v = out.reshape(b, self.latent_channels, self.Hl, self.Wl)
        return v


# ----------------------------
# Wrapper: VAE encode/decode + mask downsample
# ----------------------------
class MaskCondDiTWithVAE(nn.Module):
    """
    Key fix: latent scaling is READ from the VAE config (scaling_factor) by default.
    You can still override via latent_scale_override if you really want.
    """
    def __init__(
        self,
        dit_latent: TinyDiTLatent,
        vae_name: str = "stabilityai/sd-vae-ft-ema",
        freeze_vae: bool = True,
        latent_scale_override: Optional[float] = None,
    ):
        super().__init__()
        self.dit = dit_latent
        self.vae = AutoencoderKL.from_pretrained(vae_name)
    

        # ✅ Fix: read scaling from VAE config
        sf = getattr(self.vae.config, "scaling_factor", None)
        if sf is None:
            # very old / unusual configs: fall back to common SD value
            sf = 0.18215

        self.latent_scale = float(sf if latent_scale_override is None else latent_scale_override)
        print(f"[MaskCondDiTWithVAE] VAE={vae_name} | scaling_factor={sf} | using latent_scale={self.latent_scale}")

        if freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

    @torch.no_grad()
    def encode_image_to_latent(self, x_img: torch.Tensor) -> torch.Tensor:
        """
        diffusers AutoencoderKL expects x_img in [-1, 1]
        """
        z = self.vae.encode(x_img).latent_dist.sample()
        z = z * self.latent_scale
        return z

    @torch.no_grad()
    def decode_latent_to_image(self, z_scaled: torch.Tensor) -> torch.Tensor:
        z = z_scaled / self.latent_scale
        x = self.vae.decode(z).sample
        return x

    def downsample_mask_to_latent(self, mask_onehot: torch.Tensor) -> torch.Tensor:
        return F.interpolate(mask_onehot, size=(self.dit.Hl, self.dit.Wl), mode="nearest")

    def forward(
        self,
        z_noisy_scaled: torch.Tensor,
        t: torch.Tensor,
        mask_onehot_img: torch.Tensor,
        adaln_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask_lat = self.downsample_mask_to_latent(mask_onehot_img)
        return self.dit(z_noisy_scaled, t, mask_lat, adaln_cond=adaln_cond)