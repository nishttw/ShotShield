"""Unpack shotshield_v2_web.zip (made by notebooks/shotshield_v2_export_web.ipynb) into the demo site.

    python tools/install_web_bundle.py path/to/shotshield_v2_web.zip

  shotshield_v2.onnx, model_info.json -> docs/model/
  examples/*.wav, examples.json       -> docs/examples/
  reference.npz                       -> tools/reference/   (kept out of git)

Only these expected files are extracted; the ONNX checksum is checked against model_info.json.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = {
    "shotshield_v2.onnx": ROOT / "docs" / "model" / "shotshield_v2.onnx",
    "model_info.json": ROOT / "docs" / "model" / "model_info.json",
    "examples/examples.json": ROOT / "docs" / "examples" / "examples.json",
    "reference.npz": ROOT / "tools" / "reference" / "reference.npz",
}
EXAMPLE_WAV = re.compile(r"^examples/[A-Za-z0-9_.-]+\.wav$")


def main(zip_path: str) -> None:
    with zipfile.ZipFile(zip_path) as z:
        names = set(z.namelist())
        missing = [n for n in ("shotshield_v2.onnx", "model_info.json") if n not in names]
        if missing:
            sys.exit(f"not a ShotShield web bundle, missing: {missing}")
        info = json.loads(z.read("model_info.json"))
        onnx_bytes = z.read("shotshield_v2.onnx")
        digest = hashlib.sha256(onnx_bytes).hexdigest()
        if digest != info["web"]["onnx_sha256"]:
            sys.exit("ONNX checksum does not match model_info.json — download the zip again")
        old = ROOT / "docs" / "examples"
        for f in old.glob("*.wav"):
            f.unlink()
        for name in sorted(names):
            dest = TARGETS.get(name) or (ROOT / "docs" / name if EXAMPLE_WAV.match(name) else None)
            if dest is None:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(z.read(name))
            print(f"{name:55s} -> {dest.relative_to(ROOT)}")
    w = info["web"]
    print(f"\nONNX checksum OK ({digest[:12]}…). Converter: {w['converter']}, input {w['input_name']} "
          f"{w['input_shape']}, opset {w['opset']}; max |ONNX − Keras| on Kaggle: {w['max_abs_diff_vs_keras']:.1e} "
          f"over {w['checked_windows']} windows.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
