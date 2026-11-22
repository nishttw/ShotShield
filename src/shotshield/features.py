"""Audio -> log-mel spectrogram, exactly as in the training notebook.

If you change anything here, retrain the model: the network only understands
spectrograms made with these settings.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import librosa
import numpy as np


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 22050
    clip_seconds: float = 4.0
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    top_db: float = 80.0

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.clip_seconds)

    @property
    def n_frames(self) -> int:
        return 1 + self.n_samples // self.hop_length

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def load_audio(path: str, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """Load any audio file as mono float32 at the model's sample rate."""
    y, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    return y.astype(np.float32)


def loop_pad(y: np.ndarray, n: int) -> np.ndarray:
    """Repeat a short clip until it fills n samples; trim a long one.

    Looping (instead of padding with silence) stops the model from learning
    "short clip => gunshot", since most gunshot clips in UrbanSound8K are short.
    """
    if len(y) == 0:
        return np.zeros(n, dtype=np.float32)
    if len(y) < n:
        y = np.tile(y, int(np.ceil(n / len(y))))
    return y[:n].astype(np.float32)


def log_mel(y: np.ndarray, cfg: FeatureConfig = FeatureConfig()) -> np.ndarray:
    """One window of exactly cfg.n_samples samples -> (n_mels, n_frames) array in [0, 1]."""
    mel = librosa.feature.melspectrogram(
        y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=cfg.hop_length, n_mels=cfg.n_mels
    )
    if mel.max() < 1e-10:  # digital silence
        return np.zeros((cfg.n_mels, cfg.n_frames), dtype=np.float32)
    db = librosa.power_to_db(mel, ref=np.max, top_db=cfg.top_db)
    return ((db + cfg.top_db) / cfg.top_db).astype(np.float32)


def window_starts(n_total: int, n_window: int, n_hop: int) -> list[int]:
    """Start indices of sliding windows that cover the whole signal."""
    if n_total <= n_window:
        return [0]
    starts = list(range(0, n_total - n_window + 1, n_hop))
    if starts[-1] + n_window < n_total:
        starts.append(n_total - n_window)
    return starts


def windows_to_features(y: np.ndarray, cfg: FeatureConfig, hop_seconds: float = 1.0):
    """Split audio of any length into 4 s windows.

    Returns (features, starts): features has shape (n_windows, n_mels, n_frames, 1),
    starts holds each window's start time in seconds.
    """
    n = cfg.n_samples
    hop = max(1, int(hop_seconds * cfg.sample_rate))
    starts = window_starts(len(y), n, hop)
    feats = np.stack([log_mel(loop_pad(y[s:s + n], n), cfg) for s in starts])[..., np.newaxis]
    return feats, [s / cfg.sample_rate for s in starts]
