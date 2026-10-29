"""
omnihead: reference inference for task heads on a frozen speech backbone.

Backbone ``qwen3-asr-1.7b``: Qwen3-ASR-1.7B audio encoder (frozen, bf16). One encoder pass per clip serves every
head: language ID, speech emotion, phoneme CTC with word boundaries, end of turn, VAD and CTC ASR.

    from omnihead import OmniHead
    m = OmniHead("default")            # or "max", "full", "full_asr_ctc", ... or a path to a config json
    m.predict_files(["a.wav"])
"""
from .infer import OmniHead, load_audio
from .paths import list_configs

__version__ = "0.2.0"
__all__ = ["OmniHead", "load_audio", "list_configs", "__version__"]
