"""
Resolution of configs, head weights and backbone weights.

Environment variables
  OMNIHEAD_MODELS    root of the head weights (default: the repository's ``models/`` folder). Heads of backbone
                     ``<id>`` live under ``$OMNIHEAD_MODELS/<id>/<task>/``.
  OMNIHEAD_BACKBONE  backbone weights: a local directory or a Hugging Face id (default: the config's
                     ``encoder.model_id``, ``Qwen/Qwen3-ASR-1.7B``). A Hugging Face id is downloaded to the local
                     Hugging Face cache; the config's pinned ``encoder.revision`` is used when the id is the
                     config's own ``model_id``.
  QWEN3_ASR_SRC      optional source checkout of the ``qwen_asr`` package (otherwise pip ``qwen-asr==0.0.6``).
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

PACKAGE_DIR = Path(__file__).resolve().parent
CONFIG_DIR = PACKAGE_DIR / "configs"
DEFAULT_BACKBONE_HF_ID = "Qwen/Qwen3-ASR-1.7B"


def config_path(config: str) -> Path:
    """A shipped config name (``default``, ``full``, ...) or a path to a config json."""
    p = Path(config)
    if p.suffix == ".json" or p.is_file():
        return p
    q = CONFIG_DIR / f"{config}.json"
    if not q.exists():
        raise FileNotFoundError(f"unknown config {config!r}; shipped configs: {', '.join(list_configs())}")
    return q


def list_configs():
    return sorted(p.stem for p in CONFIG_DIR.glob("*.json"))


def models_root(override: Optional[str] = None) -> Path:
    """Head weights root: argument > $OMNIHEAD_MODELS > <repository>/models."""
    root = override or os.environ.get("OMNIHEAD_MODELS")
    if root:
        return Path(root).expanduser()
    repo_models = PACKAGE_DIR.parent / "models"
    if repo_models.is_dir():
        return repo_models
    raise FileNotFoundError("head weights not found: set OMNIHEAD_MODELS to the folder that holds "
                            "<backbone>/<task>/ (the repository's models/ folder)")


def backbone_dir(backbone: str, override: Optional[str] = None) -> Path:
    d = models_root(override) / backbone
    if not d.is_dir():
        raise FileNotFoundError(f"no head folder for backbone {backbone!r}: {d}")
    return d


def resolve_backbone_weights(encoder_cfg: dict, override: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """-> (local directory of the backbone weights, revision used or None).

    Precedence: argument > $OMNIHEAD_BACKBONE > encoder_cfg["model_id"] > Qwen/Qwen3-ASR-1.7B.
    """
    cfg_id = encoder_cfg.get("model_id", DEFAULT_BACKBONE_HF_ID)
    src = override or os.environ.get("OMNIHEAD_BACKBONE") or cfg_id
    local = Path(src).expanduser()
    if local.is_dir():
        return str(local), None
    revision = encoder_cfg.get("revision") if src == cfg_id else None
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id=src, revision=revision), revision
