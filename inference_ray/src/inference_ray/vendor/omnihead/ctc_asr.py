"""
Task kind "ctc_asr": non-autoregressive CTC ASR head on the shared frozen encoder pass (25 streams x T x 1024,
13 frames/s). Languages: en fr de es it pt (trained on those only).

Head = ``phoneme_heads.PhonemeCTCHead`` with vocab 2049: CTC blank 0, id i = SentencePiece piece i - 1
(``asr_ctc/asr_ctc_spm_unigram2048.model``). Greedy decoding: argmax per frame, collapse repeats, drop 0,
SentencePiece decode -> lowercase text without punctuation (letters incl. accents, digits, in-word apostrophes).
Clips are centre-cropped to ``max_sec`` (60 s), the same crop as the phoneme task.

    task = CtcAsrTask(cfg["tasks"]["asr_ctc"], head_dir, device="cuda:0")
    texts = task.ctc_asr(feats, valid)            # feats (25, B, T, 1024) bf16, valid (B, T) bool
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from .phoneme_heads import build_head


class CtcAsrTask:
    def __init__(self, cfg: dict, root, device="cuda:0"):
        """cfg: the task entry of a config; root: folder that ``heads`` and ``tokenizer`` are relative to."""
        import sentencepiece as spm

        root = Path(root)
        self.kind = "ctc_asr"
        self.max_sec = float(cfg.get("max_sec", 60.0))
        self.languages = list(cfg.get("languages", []))
        self.sp = spm.SentencePieceProcessor(model_file=str(root / cfg["tokenizer"]))
        self.heads = []
        for rel in cfg["heads"]:
            st = torch.load(root / rel, map_location="cpu", weights_only=False)
            h = build_head(st["config"])
            h.load_state_dict(st["state_dict"])
            self.heads.append(h.to(device).eval())

    @torch.inference_mode()
    def logprobs(self, feats, valid):
        """Mean log-softmax of the task's heads (B, T*k, V) and output lengths (B,)."""
        outs = [h(feats, valid) for h in self.heads]
        lp = torch.stack([torch.log_softmax(o[0].float(), -1) for o in outs]).mean(0)
        return lp, outs[0][1].sum(1)

    def ctc_asr(self, feats, valid) -> List[str]:
        lp, lens = self.logprobs(feats, valid)
        pred, lens = lp.argmax(-1).cpu().numpy(), lens.cpu().numpy()
        out = []
        for p, n in zip(pred, lens):
            p = p[:n]
            keep = np.ones(len(p), bool)
            keep[1:] = p[1:] != p[:-1]
            out.append(self.sp.decode([int(t) - 1 for t in p[keep] if t != 0]))
        return out
