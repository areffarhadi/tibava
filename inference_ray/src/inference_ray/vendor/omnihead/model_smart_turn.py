"""Smart-turn (end-of-turn) head on frozen Qwen3-ASR-1.7B audio-encoder taps.

Verbatim copy of the reference head in omni-mix87-sidecar-heads/model_smart_turn.py (designed
for the 32-layer, d=1280 Qwen3-Omni mix87 audio tower). The architecture is unchanged; only the
defaults are adapted to the Qwen3-ASR encoder, which exposes 25 streams of d=1024 at 13 Hz
(index 0 = input embedding after the conv front end, index k = output of encoder layer k):

    taps (N, L, 1024), packed frames at 13 Hz (L = 25 streams when all are read)
      -> LayerMix: M softmax mixes over the L tapped streams, each projected to d/M
         (a fixed per-stream standardiser is folded into the projection)
      -> right-align each clip, add a learned *frames-from-the-end* position embedding
      -> 2 pre-norm transformer layers, d=256
      -> [attention pooling ; last frame] -> MLP -> 1 logit

Reference docstring (still accurate): clips are not left-padded to 8 s as Smart Turn does;
right-aligning inside the head gives every clip the same "distance to the end of the turn"
coordinates that left-padding gives Smart Turn. ``keep`` masks taps out of the mixes;
training samples random depth prefixes so one head can be read at any depth.
``LayerProbes`` is a diagnostic: one linear probe per tap on [clip mean ; last-second mean].
"""
from __future__ import annotations

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

FRAMES_PER_SECOND = 13
MAX_FRAMES = 128  # 8 s = 104 frames at 13 Hz; headroom for clips a little over 8 s


def segment_ids(lens: torch.Tensor) -> torch.Tensor:
    return torch.repeat_interleave(torch.arange(len(lens), device=lens.device), lens)


def right_align(z: torch.Tensor, lens: torch.Tensor):
    """(N, d) packed frames -> (B, T, d) with each clip's last frame at index T-1, pad mask."""
    B, T = len(lens), int(lens.max())
    seg = segment_ids(lens)
    starts = torch.cumsum(lens, 0) - lens
    j = torch.arange(z.shape[0], device=z.device) - starts[seg]
    col = T - lens[seg] + j
    out = z.new_zeros(B, T, z.shape[-1])
    out[seg, col] = z
    pad = torch.arange(T, device=z.device)[None, :] < (T - lens)[:, None]
    return out, pad


class LayerMix(nn.Module):
    def __init__(self, num_taps: int, feature_dim: int = 1024, num_mixes: int = 4,
                 d_out: int = 256):
        super().__init__()
        assert d_out % num_mixes == 0
        self.num_mixes = num_mixes
        self.logits = nn.Parameter(torch.zeros(num_mixes, num_taps))
        self.proj = nn.Parameter(torch.empty(num_mixes, feature_dim, d_out // num_mixes))
        nn.init.normal_(self.proj, std=feature_dim ** -0.5)
        self.bias = nn.Parameter(torch.zeros(num_mixes, d_out // num_mixes))
        self.register_buffer("mean", torch.zeros(num_taps, feature_dim))
        self.register_buffer("scale", torch.ones(num_taps, feature_dim))

    def weights(self, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        lg = self.logits
        if keep is not None:
            lg = lg.masked_fill(~keep[None, :], float("-inf"))
        return lg.softmax(-1)  # (M, L)

    def forward(self, x: torch.Tensor, keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (N, L, D) raw taps -> (N, d_out). Standardise, mix over L, project per mix."""
        w = self.weights(keep)
        inv = 1.0 / self.scale
        wd = w[:, :, None] * inv[None]                         # (M, L, D)
        shift = torch.einsum("mld,ld->md", wd, self.mean)      # (M, D)
        mixed = torch.einsum("nld,mld->nmd", x, wd.to(x.dtype)) - shift.to(x.dtype)
        y = torch.einsum("nmd,mde->nme", mixed, self.proj.to(x.dtype)) + self.bias.to(x.dtype)
        return y.flatten(1)

    def importance(self) -> torch.Tensor:
        """(L,) mean softmax weight per tap across the mixes."""
        return self.weights().mean(0).detach()


class SmartTurnHead(nn.Module):
    def __init__(self, taps: Sequence[int], feature_dim: int = 1024, d_model: int = 256,
                 num_mixes: int = 4, num_layers: int = 2, num_heads: int = 4,
                 ff_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.taps = sorted(int(t) for t in taps)
        self.config = dict(taps=self.taps, feature_dim=feature_dim, d_model=d_model,
                           num_mixes=num_mixes, num_layers=num_layers, num_heads=num_heads,
                           ff_dim=ff_dim, dropout=dropout)
        self.mix = LayerMix(len(self.taps), feature_dim, num_mixes, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.pos_from_end = nn.Parameter(torch.zeros(MAX_FRAMES, d_model))
        nn.init.normal_(self.pos_from_end, std=0.02)
        block = nn.TransformerEncoderLayer(d_model, num_heads, ff_dim, dropout,
                                           activation="gelu", batch_first=True,
                                           norm_first=True)
        self.blocks = nn.TransformerEncoder(block, num_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.pool = nn.Sequential(nn.Linear(d_model, 128), nn.Tanh(), nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(2 * d_model, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    @property
    def max_layer(self) -> int:
        """Deepest tower layer this head reads (0-based); the tower can stop there."""
        return self.taps[-1]

    def forward(self, x: torch.Tensor, lens: torch.Tensor,
                keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x (N, L, D) packed taps, lens (B,) frames per clip -> (B,) logit P(turn complete)."""
        z = self.mix(x, keep)
        h, pad = right_align(z, lens)
        T = h.shape[1]
        if T > MAX_FRAMES:  # longer than the trained window: keep the end
            h, pad = h[:, -MAX_FRAMES:], pad[:, -MAX_FRAMES:]
            T = MAX_FRAMES
        h = self.in_norm(h) + self.pos_from_end[:T].flip(0).to(h.dtype)
        h = self.blocks(h, src_key_padding_mask=pad)
        h = self.out_norm(h)
        a = self.pool(h).squeeze(-1).float().masked_fill(pad, float("-inf")).softmax(1)
        pooled = torch.einsum("bt,btd->bd", a.to(h.dtype), h)
        return self.classifier(torch.cat([pooled, h[:, -1]], -1)).squeeze(-1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


class LayerProbes(nn.Module):
    """Independent linear probe per tap on standardised [clip mean ; last-second mean].

    Means are linear, so each probe is evaluated as a per-frame dot product followed by a
    segment mean, never materialising pooled features.
    """

    def __init__(self, num_taps: int, feature_dim: int = 1024, tail_frames: int = FRAMES_PER_SECOND):
        super().__init__()
        self.tail_frames = tail_frames
        self.w_all = nn.Parameter(torch.zeros(num_taps, feature_dim))
        self.w_tail = nn.Parameter(torch.zeros(num_taps, feature_dim))
        self.b = nn.Parameter(torch.zeros(num_taps))

    def forward(self, x: torch.Tensor, lens: torch.Tensor, mean: torch.Tensor,
                scale: torch.Tensor) -> torch.Tensor:
        """x (N, L, D) -> (B, L) logits."""
        B = len(lens)
        w = torch.stack([self.w_all, self.w_tail], 0) / scale[None]      # (2, L, D)
        shift = torch.einsum("kld,ld->kl", w, mean)                        # (2, L)
        s = torch.einsum("nld,kld->nkl", x, w.to(x.dtype)).float()         # (N, 2, L)
        seg = segment_ids(lens)
        starts = torch.cumsum(lens, 0) - lens
        from_end = lens[seg] - 1 - (torch.arange(x.shape[0], device=x.device) - starts[seg])
        tail = (from_end < self.tail_frames).float()
        n_tail = torch.clamp(lens, max=self.tail_frames).float()
        all_mean = torch.zeros(B, s.shape[-1], device=x.device).index_add_(0, seg, s[:, 0])
        tail_mean = torch.zeros(B, s.shape[-1], device=x.device).index_add_(
            0, seg, s[:, 1] * tail[:, None])
        all_mean = all_mean / lens[:, None].float()
        tail_mean = tail_mean / n_tail[:, None]
        return all_mean + tail_mean - shift.sum(0)[None] + self.b[None]


def load_turn_head(path: str, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    head = SmartTurnHead(**ck["head_config"])
    head.load_state_dict(ck["head"])
    return head.to(device).eval(), ck
