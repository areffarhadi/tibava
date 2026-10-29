"""Exact frame grid of the Qwen3-ASR (and Qwen3-Omni) audio encoder.

The encoder convolves every 100 mel frames (1.00 s) independently and emits 13 frames per
second. Inside a second the frames sit 80 ms apart, so the grid jumps ~40 ms at every
second boundary; a uniform 1/13 s grid would drift by up to 40 ms.

Qwen3-ASR-1.7B uses the same conv front end as the Qwen3-Omni tower (3 x conv2d,
kernel 3, stride 2, padding 1, per 100-frame chunk), so the centre offsets measured for
Omni in ``omni-mix87-sidecar-heads/omni_frontend.py`` (centre of mass of the
input-gradient of each output frame) apply unchanged.
"""
from __future__ import annotations

import numpy as np

SAMPLE_RATE = 16000
HOP = 160
MEL_CHUNK = 100
FRAMES_PER_SECOND = 13

FRAME_CENTRE_OFFSETS = np.array(
    [0.0267, 0.0949, 0.1749, 0.2549, 0.3349, 0.4149, 0.4949,
     0.5749, 0.6549, 0.7349, 0.8149, 0.8949, 0.9621],
    dtype=np.float64,
)


def num_encoder_frames(num_samples: int) -> int:
    """Encoder frames for a clip of ``num_samples`` (mirror of qwen3asr_output_lengths)."""
    mel = num_samples // HOP
    leave = mel % MEL_CHUNK
    feat = (leave - 1) // 2 + 1
    return ((feat - 1) // 2 + 1 - 1) // 2 + 1 + (mel // MEL_CHUNK) * FRAMES_PER_SECOND


def frame_centres(num_frames: int) -> np.ndarray:
    idx = np.arange(num_frames)
    return idx // FRAMES_PER_SECOND + FRAME_CENTRE_OFFSETS[idx % FRAMES_PER_SECOND]


def upsampled_frame_centres(num_frames: int, factor: int) -> np.ndarray:
    """Split each encoder frame's 80 ms cell into ``factor`` sub-cells; return their centres."""
    base = frame_centres(num_frames)
    cell = 0.08 / factor
    sub = (np.arange(factor) - (factor - 1) / 2.0) * cell
    return (base[:, None] + sub[None, :]).reshape(-1)
