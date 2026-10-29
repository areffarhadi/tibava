from typing import List, Callable, Dict
from pathlib import Path
import json

from inference_ray.plugin import AnalyserPlugin, AnalyserPluginManager
from tibava_data import AudioData, ListData, AnnotationData, Annotation
from tibava_data import DataManager, Data

import logging

# Label order of the shipped speech-emotion heads (anger … surprise).
EMOTION_LABELS = [
    "anger",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
]

SER_CONFIG = {
    "name": "ser_only",
    "backbone": "qwen3-asr-1.7b",
    "encoder": {
        "model_id": "Qwen/Qwen3-ASR-1.7B",
        "revision": "7278e1e70fe206f11671096ffdd38061171dd6e5",
        "dtype": "bfloat16",
    },
    "tasks": {
        "ser": {
            "heads": [
                "ser/s1_qformer_tf.pt",
                "ser/s1_qformer_mhfa.pt",
                "ser/s1_qformer_ecapa.pt",
            ],
            "max_sec": 15.0,
        }
    },
}

default_config = {
    "data_dir": "/data/",
    "host": "localhost",
    "port": 6379,
    "save_dir": Path("/models/audio_emotion/"),
    "models_root": "/models/omnihead",
    "label_map": EMOTION_LABELS,
}

default_parameters = {}

requires = {
    "audio": AudioData,
    "annotations": ListData,
}

provides = {
    "annotations": AnnotationData,
}


def _real_checkpoint(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 1024:
        return False
    with path.open("rb") as handle:
        return not handle.read(40).startswith(b"version https://git-lfs.github.com")


@AnalyserPluginManager.export("audio_emotion")
class AudioEmotion(
    AnalyserPlugin,
    config=default_config,
    parameters=default_parameters,
    version="0.2",
    requires=requires,
    provides=provides,
):
    def __init__(self, config=None, **kwargs):
        super().__init__(config, **kwargs)
        self.emotion_model = None

    def _load_model(self, device: str):
        from inference_ray.vendor.omnihead import OmniHead

        models_root = Path(self.config.get("models_root", "/models/omnihead"))
        head_dir = models_root / "qwen3-asr-1.7b" / "ser"
        missing = [
            name
            for name in SER_CONFIG["tasks"]["ser"]["heads"]
            if not _real_checkpoint(head_dir / Path(name).name)
        ]
        if missing:
            raise FileNotFoundError(
                "Speech-emotion heads are missing under "
                f"{head_dir}. Expected real checkpoint files, not Git LFS pointers: "
                + ", ".join(missing)
            )

        save_dir = Path(self.config.get("save_dir"))
        save_dir.mkdir(parents=True, exist_ok=True)
        config_path = save_dir / "ser_only.json"
        config_path.write_text(json.dumps(SER_CONFIG))
        return OmniHead(
            str(config_path),
            device=device,
            tasks=["ser"],
            models_root=str(models_root),
        )

    def call(
        self,
        inputs: Dict[str, Data],
        data_manager: DataManager,
        parameters: Dict = None,
        callbacks: Callable = None,
    ) -> Dict[str, Data]:
        import librosa
        import numpy as np
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.emotion_model is None:
            self.emotion_model = self._load_model(device)
        labels: List[str] = list(EMOTION_LABELS)

        def classify_segments(audio_array, speaker_turns, sampling_rate: int):
            predictions = []
            for seg in speaker_turns:
                start = int(seg.start * sampling_rate)
                end = int(seg.end * sampling_rate)
                segment = audio_array[start:end]
                if segment.size < sampling_rate:
                    segment = np.pad(segment, (0, sampling_rate - segment.size))
                wave = torch.tensor(segment, dtype=torch.float32)
                prediction = self.emotion_model.predict([wave])[0]["ser"]
                probs = [float(prediction["probs"].get(label, 0.0)) for label in labels]
                predictions.append(
                    Annotation(
                        start=seg.start,
                        end=seg.end,
                        labels=[
                            {
                                "emotion_probs": probs,
                                "emotion_pred": prediction["label"],
                            }
                        ],
                    )
                )
            return predictions

        with (
            inputs["audio"] as input_audio,
            inputs["annotations"] as input_annotations,
            data_manager.create_data("AnnotationData") as annotation_data,
        ):
            with input_audio.open_audio("r") as audio_file:
                sampling_rate = 16000
                audio_array, _ = librosa.load(audio_file, sr=sampling_rate, mono=True)
                for _, speaker_data in input_annotations:
                    with speaker_data as speaker_data:
                        annotation_data.annotations.extend(
                            classify_segments(
                                audio_array, speaker_data.annotations, sampling_rate
                            )
                        )

                self.update_callbacks(callbacks, progress=1.0)
                return {"annotations": annotation_data}
