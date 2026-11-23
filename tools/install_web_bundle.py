"""Unpack a ShotShield web bundle (shotshield_v*_web.zip, made on Kaggle) into the demo site.

    python tools/install_web_bundle.py path/to/shotshield_v4_web.zip [--threshold 0.80]

  <model>.onnx, model_info.json       -> docs/model/   (older .onnx files there are removed)
  examples/*.wav, examples.json       -> docs/examples/
  reference.npz                       -> tools/reference/   (kept out of git)

Only these expected files are extracted; the ONNX checksum is checked against model_info.json.
--threshold replaces the notebook's threshold in model_info.json and examples.json (the notebook's value is
kept as "threshold_notebook") and re-marks which example clips count as detected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "docs" / "model"
TARGETS = {
    "model_info.json": ROOT / "docs" / "model" / "model_info.json",
    "examples/examples.json": ROOT / "docs" / "examples" / "examples.json",
    "reference.npz": ROOT / "tools" / "reference" / "reference.npz",
}
EXAMPLE_WAV = re.compile(r"^examples/[A-Za-z0-9_.-]+\.wav$")


def set_threshold(threshold: float, note: str) -> None:
    info_path, ex_path = TARGETS["model_info.json"], TARGETS["examples/examples.json"]
    info = json.loads(info_path.read_text(encoding="utf-8"))
    info["threshold_notebook"] = info.get("threshold_notebook", info["threshold"])
    info["threshold"] = threshold
    info["threshold_note"] = note
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    ex = json.loads(ex_path.read_text(encoding="utf-8"))
    ex["threshold"] = threshold
    for e in ex["examples"]:
        e["keras_detected"] = e["keras_max_prob"] >= threshold
    ex_path.write_text(json.dumps(ex, indent=2), encoding="utf-8")
    print(f"threshold {info['threshold_notebook']:.3f} (notebook) -> {threshold:.3f}")


def main(zip_path: str, threshold: float | None, note: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        if "model_info.json" not in names:
            sys.exit("not a ShotShield web bundle: model_info.json missing")
        info = json.loads(z.read("model_info.json"))
        onnx_name = info["web"]["onnx_file"]
        if not re.fullmatch(r"[A-Za-z0-9_.-]+\.onnx", onnx_name) or onnx_name not in names:
            sys.exit(f"model file {onnx_name!r} missing from the bundle")
        TARGETS[onnx_name] = MODEL_DIR / onnx_name
        onnx_bytes = z.read(onnx_name)
        digest = hashlib.sha256(onnx_bytes).hexdigest()
        if digest != info["web"]["onnx_sha256"]:
            sys.exit("ONNX checksum does not match model_info.json — download the zip again")
        for f in (ROOT / "docs" / "examples").glob("*.wav"):
            f.unlink()
        for f in MODEL_DIR.glob("*.onnx"):
            if f.name != onnx_name:
                f.unlink()
        for name in sorted(names):
            dest = TARGETS.get(name) or (ROOT / "docs" / name if EXAMPLE_WAV.match(name) else None)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            print(f"{name:55s} -> {dest.relative_to(ROOT)}")
    w = info["web"]
    print(f"\nONNX checksum OK ({digest[:12]}...). Converter: {w['converter']}, input {w['input_name']} "
          f"{w['input_shape']}, opset {w['opset']}; max |ONNX - Keras| on Kaggle: {w['max_abs_diff_vs_keras']:.1e} "
          f"over {w['checked_windows']} windows.")
    if threshold is not None:
        set_threshold(threshold, note)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("zip")
    ap.add_argument("--threshold", type=float, help="decision threshold to ship instead of the notebook's")
    ap.add_argument("--note", default="set by hand when installing", help="why the threshold was changed")
    a = ap.parse_args()
    if a.threshold is not None and not 0 < a.threshold < 1:
        sys.exit("--threshold must be between 0 and 1")
    main(a.zip, a.threshold, a.note)
