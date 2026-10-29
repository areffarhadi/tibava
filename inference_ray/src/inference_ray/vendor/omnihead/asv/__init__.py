"""OmniASV: speaker-embedding heads on the frozen Qwen3-ASR-1.7B encoder (backbone ``qwen3-asr-1.7b``).

Two heads: ``4s`` (trained on 4 s crops; best on full utterances) and ``var`` (trained on 0.3-6 s speech-only
crops; for short segments, e.g. diarization). See docs/ASV.md.

    from omnihead.asv import SpeakerEmbedder
    e = SpeakerEmbedder("var")
    print(e.score("a.wav", "b.wav"))
"""
from .model import OmniASVHead, build_from_checkpoint


def __getattr__(name):
    if name in ("SpeakerEmbedder", "load_head", "HEADS"):
        from . import embedder
        return getattr(embedder, name)
    raise AttributeError(name)


__all__ = ["OmniASVHead", "build_from_checkpoint", "SpeakerEmbedder", "load_head"]
