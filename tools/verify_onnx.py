"""End-to-end check of the browser pipeline against the original Keras model.

For every waveform in tools/reference/reference.npz (made on Kaggle):
    JavaScript features (docs/js/features.js, run in Node) -> ONNX model (docs/model/) -> probability
is compared with
    librosa features -> Keras model (probabilities saved on Kaggle).

    python tools/verify_onnx.py [--reference tools/reference/reference.npz]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from verify_features import run_js  # noqa: E402

TOLERANCE = 1e-4


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, default=ROOT / "tools" / "reference" / "reference.npz")
    args = ap.parse_args()

    info = json.loads((ROOT / "docs" / "model" / "model_info.json").read_text())
    threshold = info["threshold"]
    sess = ort.InferenceSession(str(ROOT / "docs" / "model" / "shotshield_v2.onnx"), providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    npz = np.load(args.reference, allow_pickle=False)
    names = [k[5:] for k in npz.files if k.startswith("wave/")]
    js = run_js({n: npz[f"wave/{n}"].astype(np.float32) for n in names})

    def onnx_prob(feats):  # feats (n_windows, 128, 173) -> (n_windows,)
        return np.array([sess.run(None, {inp.name: f[None, :, :, None].astype(np.float32)})[0].ravel()[0] for f in feats])

    print(f"ONNX input {inp.name} {inp.shape}; threshold {threshold:.3f}\n")
    print(f"{'signal':34s} {'win':>3s} {'Keras':>7s} {'browser':>8s} {'|diff| py-feat':>14s} {'|diff| js-feat':>14s}  result")
    worst, ok_all = 0.0, True
    for n in names:
        p_keras = npz[f"prob/{n}"]
        p_py = onnx_prob(npz[f"feat/{n}"][..., 0])
        p_js = onnx_prob(js[n][0])
        d_py, d_js = np.abs(p_py - p_keras).max(), np.abs(p_js - p_keras).max()
        same = np.array_equal(p_js >= threshold, p_keras >= threshold)
        ok = d_js <= TOLERANCE and d_py <= TOLERANCE and same
        ok_all &= ok
        worst = max(worst, d_js)
        print(f"{n:34s} {len(p_keras):3d} {p_keras.max():7.4f} {p_js.max():8.4f} {d_py:14.2e} {d_js:14.2e}  "
              f"{'OK' if ok else 'FAIL'}")
    print(f"\nworst |browser pipeline − Keras|: {worst:.2e} (tolerance {TOLERANCE:.0e})")
    print("ALL OK" if ok_all else "MISMATCH")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
