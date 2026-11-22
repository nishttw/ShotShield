"""Check that the browser's log-mel front-end (docs/js/features.js) matches librosa numerically.

Runs the JavaScript code in Node on a set of test signals and compares every value with
src/shotshield/features.py (the code the model was trained with).

    python tools/verify_features.py                         # against the local librosa
    python tools/verify_features.py --reference ref.npz     # also against references made on Kaggle

ref.npz comes from notebooks/shotshield_v2_export.ipynb: waveforms ("wave/<name>") and the features
librosa computed for them on Kaggle ("feat/<name>", one row per 4 s window), with the librosa
version the model was trained with.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from shotshield.features import FeatureConfig, windows_to_features  # noqa: E402

CFG = FeatureConfig()
SR = CFG.sample_rate
# Features are in [0, 1] where 1.0 = 80 dB, so 1e-4 is 0.008 dB.
TOLERANCE = 1e-4


def test_signals() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    t = np.arange(CFG.n_samples) / SR

    def burst(n, decay):
        return (rng.standard_normal(n) * np.exp(-np.arange(n) / (decay * SR))).astype(np.float32)

    shots = np.zeros(CFG.n_samples, np.float32)
    for start in (0.3, 1.1, 1.25, 2.9):
        i = int(start * SR)
        shots[i:i + SR // 2] += 0.9 * burst(SR // 2, 0.03)
    background = 0.01 * rng.standard_normal(CFG.n_samples)
    long_mix = np.concatenate([background, shots + background, 0.2 * np.sin(2 * np.pi * 440 * t)])[: int(9.3 * SR)]

    signals = {
        "white_noise": 0.3 * rng.standard_normal(CFG.n_samples),
        "sine_1khz": 0.5 * np.sin(2 * np.pi * 1000 * t),
        "chirp_50hz_10khz": 0.5 * np.sin(2 * np.pi * 50 * 4 / np.log(200) * (200 ** (t / 4) - 1)),
        "impulses_in_noise": shots + background,
        "single_click": np.where(np.arange(CFG.n_samples) == 40000, 1.0, 0.0),
        "short_clip_0.7s_looped": burst(int(0.7 * SR), 0.1),
        "short_clip_1_sample": np.array([0.5]),
        "long_mix_9.3s": long_mix,
        "square_full_scale": np.sign(np.sin(2 * np.pi * 220 * t)),
        "dc_plus_hum": 0.5 + 0.1 * np.sin(2 * np.pi * 50 * t),
        "quiet_noise_1e-4": 1e-4 * rng.standard_normal(CFG.n_samples),
        "digital_silence": np.zeros(CFG.n_samples),
        "empty": np.zeros(0),
    }
    return {k: v.astype(np.float32) for k, v in signals.items()}


def run_js(signals: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, list[float], float]]:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        manifest = []
        for i, (name, y) in enumerate(signals.items()):
            f = f"sig{i}.f32"
            y.astype("<f4").tofile(tmp / f)
            manifest.append({"name": f"sig{i}", "file": f, "hop_seconds": 1.0})
        (tmp / "manifest.json").write_text(json.dumps(manifest))
        subprocess.run(["node", str(ROOT / "tools" / "features_cli.mjs"), str(tmp)], check=True)
        out = {}
        for i, name in enumerate(signals):
            feats = np.fromfile(tmp / f"sig{i}.out.f32", dtype="<f4").reshape(-1, CFG.n_mels, CFG.n_frames)
            meta = json.loads((tmp / f"sig{i}.starts.json").read_text())
            out[name] = (feats, meta["starts"], meta["ms_per_window"])
        return out


def compare(label: str, signals, js, expected) -> bool:
    print(f"\n=== JavaScript vs {label} ===")
    print(f"{'signal':26s} {'windows':>7s} {'max |diff|':>11s} {'mean |diff|':>12s} {'max dB':>8s}  result")
    ok_all = True
    worst = 0.0
    for name in signals:
        feats_js, starts_js, _ = js[name]
        feats_py, starts_py = expected[name]
        ok = feats_js.shape == feats_py.shape and np.allclose(starts_js, starts_py)
        diff = np.abs(feats_js - feats_py) if ok else np.array([np.inf])
        ok = ok and diff.max() <= TOLERANCE
        worst = max(worst, float(diff.max()))
        ok_all &= ok
        print(f"{name:26s} {len(feats_py):7d} {diff.max():11.2e} {diff.mean():12.2e} {80 * diff.max():8.5f}  "
              f"{'OK' if ok else 'FAIL'}")
    print(f"worst difference: {worst:.2e} (= {80 * worst:.5f} dB); tolerance {TOLERANCE:.0e}")
    return ok_all


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", type=Path, help="reference.npz made on Kaggle")
    args = ap.parse_args()

    signals = test_signals()
    expected = {name: (f[..., 0], s) for name, y in signals.items() for f, s in [windows_to_features(y, CFG)]}
    ref = None
    if args.reference:
        npz = np.load(args.reference, allow_pickle=False)
        ref_info = json.loads(str(npz["info"]))
        ref = {}
        for key in npz.files:
            if key.startswith("wave/"):
                name = key[5:]
                signals[name] = npz[key].astype(np.float32)
                ref[name] = (npz[f"feat/{name}"][..., 0] if npz[f"feat/{name}"].ndim == 4 else npz[f"feat/{name}"],
                             list(npz[f"starts/{name}"]))
                f, s = windows_to_features(signals[name], CFG)
                expected[name] = (f[..., 0], s)

    js = run_js(signals)
    ms = np.mean([v[2] for v in js.values()])
    print(f"JavaScript (Node {subprocess.run(['node', '--version'], capture_output=True, text=True).stdout.strip()}): "
          f"{ms:.1f} ms per 4 s window")

    ok = compare(f"local librosa {librosa.__version__}", signals, js, expected)
    if ref is not None:
        ok &= compare(f"Kaggle librosa {ref_info['librosa']} (training version)", list(ref), js, ref)
    print("\nALL OK" if ok else "\nMISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
