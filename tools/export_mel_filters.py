"""Export librosa's mel filterbank as sparse JSON for the browser demo (docs/model/mel_filters.json).

The JavaScript front-end (docs/js/features.js) multiplies by these exact weights instead of
recomputing them, so the filterbank is identical to the one used in training.

    python tools/export_mel_filters.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotshield.features import FeatureConfig  # noqa: E402

OUT = ROOT / "docs" / "model" / "mel_filters.json"


def main() -> None:
    cfg = FeatureConfig()
    # Same call (and defaults) that librosa.feature.melspectrogram makes internally
    basis = librosa.filters.mel(sr=cfg.sample_rate, n_fft=cfg.n_fft, n_mels=cfg.n_mels)
    filters = []
    for row in basis:
        nz = np.flatnonzero(row)
        start, stop = (int(nz[0]), int(nz[-1]) + 1) if len(nz) else (0, 0)
        # '%.9g' round-trips float32 exactly
        filters.append({"start": start, "weights": [float("%.9g" % w) for w in row[start:stop]]})
    data = {
        "sample_rate": cfg.sample_rate,
        "n_fft": cfg.n_fft,
        "n_mels": cfg.n_mels,
        "librosa_version": librosa.__version__,
        "note": "librosa.filters.mel(sr, n_fft, n_mels) with default fmin=0, fmax=sr/2, htk=False, norm='slaney'",
        "filters": filters,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")))
    rebuilt = np.zeros_like(basis)
    for m, f in enumerate(filters):
        rebuilt[m, f["start"]:f["start"] + len(f["weights"])] = f["weights"]
    assert np.array_equal(rebuilt, basis), "sparse export does not reproduce the filterbank"
    print(f"wrote {OUT.relative_to(ROOT)}: {sum(len(f['weights']) for f in filters)} non-zero weights, "
          f"{OUT.stat().st_size / 1024:.1f} KB, librosa {librosa.__version__}")


if __name__ == "__main__":
    main()
