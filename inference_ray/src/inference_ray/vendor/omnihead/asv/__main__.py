"""CLI.
  python -m omnihead.asv embed a.wav b.wav ... --out emb.npz [--head var|4s]
  python -m omnihead.asv score a.wav b.wav [--head var|4s]
  python -m omnihead.asv trials trials.txt --root <wav root> --out scores.txt   (lines: label path1 path2)
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser(description="OmniASV speaker embeddings (backbone qwen3-asr-1.7b)")
    ap.add_argument("mode", choices=["embed", "score", "trials"])
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--head", default="var")
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default="")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    from .embedder import SpeakerEmbedder
    e = SpeakerEmbedder(a.head, a.device)
    if a.mode == "embed":
        emb = e.embed_files(a.inputs)
        np.savez(a.out or "embeddings.npz", paths=np.array(a.inputs), emb=emb)
        print(f"{len(a.inputs)} embeddings ({emb.shape[1]}-d) -> {a.out or 'embeddings.npz'}")
    elif a.mode == "score":
        print(f"{e.score(a.inputs[0], a.inputs[1]):.4f}")
    else:
        trials = [l.split() for l in open(a.inputs[0]) if l.strip()]
        files = sorted({t[1] for t in trials} | {t[2] for t in trials})
        emb = e.embed_files([os.path.join(a.root, f) for f in files])
        idx = {f: i for i, f in enumerate(files)}
        with open(a.out or "scores.txt", "w") as f:
            for t in trials:
                f.write(f"{t[0]} {t[1]} {t[2]} {float(emb[idx[t[1]]] @ emb[idx[t[2]]]):.5f}\n")
        print(f"{len(trials)} trials scored -> {a.out or 'scores.txt'}")


if __name__ == "__main__":
    main()
