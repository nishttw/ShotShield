"""Check an audio file for gunshots.

    python predict.py path/to/recording.wav
    python predict.py recording.mp3 --hop 0.5 --json
"""
import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shotshield import ShotShieldDetector  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Detect gunshots in an audio file.")
    parser.add_argument("audio", help="audio file (wav, mp3, flac, ogg, ...)")
    parser.add_argument("--model", default=ROOT / "models" / "shotshield_v4.keras")
    parser.add_argument("--info", default=ROOT / "models" / "model_info.json")
    parser.add_argument("--hop", type=float, default=1.0, help="seconds between 4 s windows (default 1.0)")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()

    for p in (args.model, args.info):
        if not Path(p).exists():
            sys.exit(f"Missing {p}. Download the model from the GitHub Releases page into models/ (see README).")

    detector = ShotShieldDetector.load(args.model, args.info)
    result = detector.predict_file(args.audio, hop_seconds=args.hop)

    if args.json:
        print(json.dumps({**asdict(result), "gunshot_detected": result.gunshot_detected,
                          "max_probability": result.max_probability}, indent=2))
        return 0

    print(f"{Path(args.audio).name}: {result.duration:.1f} s, {len(result.windows)} window(s)")
    if result.gunshot_detected:
        print(f"GUNSHOT DETECTED (threshold {result.threshold:.2f})")
        for d in result.detections:
            print(f"  {d.start:6.1f}s – {d.end:6.1f}s   peak probability {d.peak_probability:.1%}")
    else:
        print(f"No gunshot detected (highest probability {result.max_probability:.1%}, "
              f"threshold {result.threshold:.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
