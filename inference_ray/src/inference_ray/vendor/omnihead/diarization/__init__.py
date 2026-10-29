"""Speaker diarization head on the frozen Qwen3-ASR-1.7B encoder (backbone ``qwen3-asr-1.7b``).

    from omnihead.diarization import Diarizer
    d = Diarizer(embedding="/path/to/wespeaker_samresnet34_voxblink2.onnx")
    print(d("meeting.wav").to_rttm())

The head itself (``QwenDiarHead``) needs only torch; the full-recording pipeline (``Diarizer``)
needs the DiariZen fork of pyannote.audio and onnxruntime (requirements-diar.txt).
See docs/DIARIZATION.md.
"""
from .model import QwenDiarHead, build_from_checkpoint, powerset_mapping


def __getattr__(name):
    if name in ("Diarizer", "encode_long", "default_checkpoint"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)


__all__ = ["QwenDiarHead", "build_from_checkpoint", "powerset_mapping", "Diarizer"]
