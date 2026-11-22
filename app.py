"""ShotShield web demo: upload or record audio and see where gunshots occur.

    python app.py            # then open http://127.0.0.1:7860
"""
import sys
from pathlib import Path

import gradio as gr
import librosa
import librosa.display
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from shotshield import ShotShieldDetector  # noqa: E402
from shotshield.features import load_audio  # noqa: E402

MAX_SECONDS = 120  # keep the demo responsive
detector = ShotShieldDetector.load(ROOT / "models" / "shotshield_v3.keras", ROOT / "models" / "model_info.json")
cfg = detector.cfg


def plot(y, result):
    fig, (ax_spec, ax_prob) = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True,
                                           gridspec_kw={"height_ratios": [3, 2]})
    mel = librosa.feature.melspectrogram(y=y, sr=cfg.sample_rate, n_fft=cfg.n_fft,
                                         hop_length=cfg.hop_length, n_mels=cfg.n_mels)
    librosa.display.specshow(librosa.power_to_db(mel, ref=mel.max() or 1.0), sr=cfg.sample_rate,
                             hop_length=cfg.hop_length, x_axis="time", y_axis="mel", ax=ax_spec, cmap="magma")
    ax_spec.set_title("Mel spectrogram")
    ax_spec.set_xlabel("")

    centres = [(w.start + w.end) / 2 for w in result.windows]
    ax_prob.plot(centres, [w.probability for w in result.windows], marker="o", color="#d62728")
    ax_prob.axhline(result.threshold, ls="--", color="k", lw=1, label=f"threshold {result.threshold:.2f}")
    for d in result.detections:
        ax_spec.axvspan(d.start, d.end, color="#d62728", alpha=0.25)
        ax_prob.axvspan(d.start, d.end, color="#d62728", alpha=0.15)
    ax_prob.set_ylim(0, 1.02)
    ax_prob.set_xlim(0, max(result.duration, 0.1))
    ax_prob.set_ylabel("gunshot probability")
    ax_prob.set_xlabel("time (s)")
    ax_prob.legend(loc="upper right")
    fig.tight_layout()
    return fig


def analyze(audio_path):
    if not audio_path:
        return None, None, "Upload or record some audio first."
    y = load_audio(audio_path, cfg)[: MAX_SECONDS * cfg.sample_rate]
    result = detector.predict_waveform(y, hop_seconds=1.0)
    p = result.max_probability
    if result.gunshot_detected:
        times = ", ".join(f"{d.start:.1f}–{d.end:.1f} s" for d in result.detections)
        summary = f"### 🔴 Gunshot detected\nAround: {times}"
    else:
        summary = "### 🟢 No gunshot detected"
    summary += f"\n\nAnalysed {result.duration:.1f} s in {len(result.windows)} window(s) of 4 s."
    return {"Gunshot": p, "No gunshot": 1 - p}, plot(y, result), summary


demo = gr.Interface(
    fn=analyze,
    inputs=gr.Audio(type="filepath", sources=["upload", "microphone"], label="Audio"),
    outputs=[gr.Label(label="Highest gunshot probability"), gr.Plot(label="Where in the audio"),
             gr.Markdown()],
    title="ShotShield — gunshot detection",
    description=("A CNN trained on UrbanSound8K listens in 4-second windows and estimates the probability "
                 "that each one contains a gunshot. Research demo only — not a safety system."),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
