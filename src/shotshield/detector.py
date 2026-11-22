"""Load the trained model and find gunshots in audio of any length."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .features import FeatureConfig, load_audio, windows_to_features


@dataclass
class Window:
    start: float  # seconds
    end: float
    probability: float


@dataclass
class Detection:
    start: float
    end: float
    peak_probability: float


@dataclass
class Result:
    duration: float
    threshold: float
    windows: list[Window] = field(default_factory=list)
    detections: list[Detection] = field(default_factory=list)

    @property
    def max_probability(self) -> float:
        return max((w.probability for w in self.windows), default=0.0)

    @property
    def gunshot_detected(self) -> bool:
        return bool(self.detections)


class ShotShieldDetector:
    def __init__(self, model, info: dict):
        self.model = model
        self.info = info
        self.cfg = FeatureConfig.from_dict(info["features"])
        self.threshold = float(info.get("threshold", 0.5))

    @classmethod
    def load(cls, model_path: str | Path, info_path: str | Path) -> "ShotShieldDetector":
        import keras  # imported here so the feature code works without TensorFlow

        info = json.loads(Path(info_path).read_text())
        return cls(keras.models.load_model(model_path), info)

    def predict_waveform(self, y: np.ndarray, hop_seconds: float = 1.0) -> Result:
        """y: mono audio at self.cfg.sample_rate."""
        feats, starts = windows_to_features(y, self.cfg, hop_seconds)
        probs = self.model.predict(feats, batch_size=32, verbose=0).ravel()
        duration = len(y) / self.cfg.sample_rate
        windows = [
            Window(s, min(s + self.cfg.clip_seconds, duration), float(p)) for s, p in zip(starts, probs)
        ]
        return Result(duration, self.threshold, windows, self._merge(windows))

    def predict_file(self, path: str | Path, hop_seconds: float = 1.0) -> Result:
        return self.predict_waveform(load_audio(str(path), self.cfg), hop_seconds)

    def _merge(self, windows: list[Window]) -> list[Detection]:
        """Join overlapping windows above the threshold into detection intervals."""
        events: list[Detection] = []
        for w in windows:
            if w.probability < self.threshold:
                continue
            if events and w.start <= events[-1].end:
                events[-1].end = max(events[-1].end, w.end)
                events[-1].peak_probability = max(events[-1].peak_probability, w.probability)
            else:
                events.append(Detection(w.start, w.end, w.probability))
        return events
