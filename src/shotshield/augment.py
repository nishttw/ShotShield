"""Training-time audio augmentation for ShotShield v4.

The live demo hears sounds through a chain the clean datasets never contain: a loudspeaker (for
played-back clips), a room, a cheap microphone and sometimes a lossy codec. These functions
simulate that chain. They are applied to gunshots *and* to other sounds alike, so the model
cannot learn "sounds like a speaker => gunshot".

Every function takes a numpy random Generator, so an augmented copy is reproducible from its seed.
"""
from __future__ import annotations

import shutil
import subprocess

import numpy as np
import scipy.signal as sps

PINK_B = [0.049922035, -0.095993537, 0.050612699, -0.004408786]
PINK_A = [1, -2.494956002, 2.017265875, -0.522189400]
FFMPEG = shutil.which("ffmpeg")


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x, dtype=np.float64)))) + 1e-12


def peaking_eq(y, sr, f0, gain_db, q):
    """RBJ peaking filter: a resonance (gain_db > 0) or a dip at f0."""
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = [1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a]
    den = [1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a]
    return sps.lfilter(np.array(b) / den[0], np.array(den) / den[0], y)


def butter(y, sr, fc, kind, order=2):
    fc = min(fc, 0.45 * sr)
    return sps.sosfilt(sps.butter(order, fc, kind, fs=sr, output="sos"), y)


def room(y, sr, rng, rt60=None, drr_db=None):
    """Synthetic room: a few early reflections plus an exponentially decaying diffuse tail."""
    rt60 = rng.uniform(0.15, 0.9) if rt60 is None else rt60
    drr_db = rng.uniform(-6, 12) if drr_db is None else drr_db   # direct-to-reverberant ratio
    n = int(rt60 * sr)
    t = np.arange(n) / sr
    tail = rng.standard_normal(n) * np.exp(-6.9 * t / rt60)
    tail = butter(tail, sr, rng.uniform(3000, 9000), "lowpass")  # walls absorb treble
    for _ in range(rng.integers(3, 8)):                          # early reflections, 2-30 ms
        k = int(rng.uniform(0.002, 0.03) * sr)
        if k < n:
            tail[k] += rng.uniform(0.3, 0.8) * rng.choice([-1, 1])
    tail[0] = 0
    wet = sps.fftconvolve(y, tail)[: len(y)]
    wet *= rms(y) / rms(wet) * 10 ** (-drr_db / 20)
    return y + wet


def loudspeaker(y, sr, rng):
    """Small loudspeaker (phone, laptop, cheap Bluetooth): no deep bass, a resonance, compression."""
    # cut-off range chosen so the bass loss matches 27 YouTube gunshots recorded through a laptop
    # (energy below 150 Hz 20-46 dB under the total)
    y = butter(y, sr, rng.uniform(80, 500), "highpass", order=4)
    y = butter(y, sr, rng.uniform(5000, 11000), "lowpass", order=4)
    y = peaking_eq(y, sr, rng.uniform(800, 4000), rng.uniform(2, 9), rng.uniform(0.7, 3))
    drive = rng.uniform(1, 6)                                    # limiter / overdriven amplifier
    peak = np.max(np.abs(y)) + 1e-12
    return np.tanh(drive * y / peak) * peak / np.tanh(drive)


def microphone(y, sr, rng):
    """Built-in laptop/phone microphone: rolled-off bass, uneven mids, sometimes dull treble."""
    y = butter(y, sr, rng.uniform(50, 250), "highpass")
    y = peaking_eq(y, sr, rng.uniform(1500, 6000), rng.uniform(-4, 6), rng.uniform(0.5, 2))
    if rng.random() < 0.4:
        y = butter(y, sr, rng.uniform(6000, 10000), "lowpass")
    return y


def codec(y, sr, rng):
    """Lossy compression (MP3 at a low bit rate) as in streamed video. Needs ffmpeg; else unchanged."""
    if FFMPEG is None:
        return y
    rate = str(int(rng.choice([32, 48, 64, 96, 128]))) + "k"
    pcm = np.clip(y / (np.max(np.abs(y)) + 1e-12) * 0.9, -1, 1).astype("<f4").tobytes()
    fmt = ["-f", "f32le", "-ar", str(sr), "-ac", "1"]
    try:
        mp3 = subprocess.run([FFMPEG, "-loglevel", "error", *fmt, "-i", "pipe:0", "-c:a", "libmp3lame", "-b:a", rate,
                              "-f", "mp3", "pipe:1"], input=pcm, capture_output=True, check=True, timeout=60).stdout
        out = subprocess.run([FFMPEG, "-loglevel", "error", "-i", "pipe:0", *fmt, "pipe:1"], input=mp3,
                             capture_output=True, check=True, timeout=60).stdout
    except (subprocess.SubprocessError, OSError):
        return y
    z = np.frombuffer(out, "<f4").astype(np.float64)
    delay = int(np.argmax(np.abs(sps.correlate(z[: len(y) + 4096], y, mode="valid")))) if len(z) > len(y) else 0
    z = z[delay: delay + len(y)]
    if len(z) < len(y):
        z = np.pad(z, (0, len(y) - len(z)))
    return z * (rms(y) / rms(z))


def add_noise(y, rng, snr_db):
    noise = rng.standard_normal(len(y))
    if rng.random() < 0.5:
        noise = sps.lfilter(PINK_B, PINK_A, noise)
    return y + noise * (rms(y) / rms(noise)) * 10 ** (-snr_db / 20)


def add_background(y, bg, rng, snr_db=None):
    snr_db = rng.uniform(0, 20) if snr_db is None else snr_db
    return y + bg * (rms(y) / rms(bg)) * 10 ** (-snr_db / 20)


def soft_clip(y, rng):
    g = rng.uniform(2, 10)
    return np.tanh(g * y / (np.max(np.abs(y)) + 1e-12))


def place_event(event, n, rng, bg=None, snr_db=None, margin=0.1):
    """Put a short sound at a random position inside an n-sample window (over a background, if given).

    Live windows slide past a sound, so it can sit anywhere in the 4 s, not only in the middle.
    """
    event = event[:n]
    out = np.zeros(n) if bg is None else np.array(bg[:n], dtype=np.float64)
    lo = int(-margin * len(event))                      # the sound may be cut a little at either edge
    hi = n - int((1 - margin) * len(event))
    start = int(rng.integers(lo, max(lo + 1, hi)))
    a, b = max(0, start), min(n, start + len(event))
    piece = event[a - start: b - start]
    if bg is not None:
        level = rms(piece) * 10 ** (-(rng.uniform(5, 30) if snr_db is None else snr_db) / 20)
        out *= level / rms(out)
    out[a:b] += piece
    return out


def playback_chain(y, sr, rng, use_codec=True):
    """A clip played through a loudspeaker in a room and recorded by a built-in microphone."""
    if use_codec and rng.random() < 0.5:
        y = codec(y, sr, rng)
    y = loudspeaker(y, sr, rng)
    if rng.random() < 0.8:
        y = room(y, sr, rng)
    return microphone(y, sr, rng)


def augment(y, sr, rng, bg=None, p_playback=0.5, use_codec=True):
    """One augmented copy: either the playback chain or a direct room/mic recording, then noise etc."""
    y = np.asarray(y, dtype=np.float64)
    if rng.random() < p_playback:
        y = playback_chain(y, sr, rng, use_codec)
    else:
        if rng.random() < 0.5:
            y = room(y, sr, rng)
        if rng.random() < 0.5:
            y = microphone(y, sr, rng)
    if bg is not None:
        y = add_background(y, bg, rng)
    if rng.random() < 0.3:
        y = add_noise(y, rng, rng.uniform(15, 45))
    if rng.random() < 0.15:
        y = soft_clip(y, rng)
    return y.astype(np.float32)
