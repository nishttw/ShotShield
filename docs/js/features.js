// Audio -> log-mel spectrogram, a JavaScript port of src/shotshield/features.py.
//
// It reproduces librosa.feature.melspectrogram(y, sr=22050, n_fft=2048, hop_length=512, n_mels=128)
// with librosa's defaults (periodic Hann window, center=True, zero padding, power 2, Slaney mel filters)
// followed by power_to_db(ref=np.max, top_db=80) and scaling to [0, 1].
// The mel filterbank is not recomputed here: it is exported from librosa (tools/export_mel_filters.py)
// so both sides multiply by exactly the same weights.
//
// Checked against librosa by tools/verify_features.py. No DOM code, so it runs in the browser and in Node.

export const DEFAULT_CONFIG = Object.freeze({
  sampleRate: 22050,
  clipSeconds: 4.0,
  nMels: 128,
  nFft: 2048,
  hopLength: 512,
  topDb: 80.0,
});

export function configFromInfo(features) {
  return {
    sampleRate: features.sample_rate,
    clipSeconds: features.clip_seconds,
    nMels: features.n_mels,
    nFft: features.n_fft,
    hopLength: features.hop_length,
    topDb: features.top_db,
  };
}

export const nSamples = (cfg) => Math.trunc(cfg.sampleRate * cfg.clipSeconds);
export const nFrames = (cfg) => 1 + Math.floor(nSamples(cfg) / cfg.hopLength);

/** Repeat a short clip until it fills n samples; trim a long one (features.loop_pad). */
export function loopPad(y, n) {
  const out = new Float32Array(n);
  if (y.length === 0) return out;
  if (y.length >= n) {
    out.set(y.subarray(0, n));
    return out;
  }
  for (let i = 0; i < n; i += y.length) out.set(y.subarray(0, Math.min(y.length, n - i)), i);
  return out;
}

/** Start indices of sliding windows that cover the whole signal (features.window_starts). */
export function windowStarts(nTotal, nWindow, nHop) {
  if (nTotal <= nWindow) return [0];
  const starts = [];
  for (let s = 0; s <= nTotal - nWindow; s += nHop) starts.push(s);
  if (starts[starts.length - 1] + nWindow < nTotal) starts.push(nTotal - nWindow);
  return starts;
}

/** In-place radix-2 complex FFT with precomputed tables. */
class FFT {
  constructor(n) {
    if (n & (n - 1)) throw new Error('FFT size must be a power of two');
    this.n = n;
    this.rev = new Uint32Array(n);
    const bits = Math.log2(n);
    for (let i = 0; i < n; i++) {
      let r = 0;
      for (let b = 0; b < bits; b++) r |= ((i >> b) & 1) << (bits - 1 - b);
      this.rev[i] = r;
    }
    this.cos = new Float64Array(n / 2);
    this.sin = new Float64Array(n / 2);
    for (let i = 0; i < n / 2; i++) {
      this.cos[i] = Math.cos((-2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((-2 * Math.PI * i) / n);
    }
  }

  transform(re, im) {
    const n = this.n;
    for (let i = 0; i < n; i++) {
      const j = this.rev[i];
      if (j > i) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let size = 2; size <= n; size <<= 1) {
      const half = size >> 1;
      const step = n / size;
      for (let start = 0; start < n; start += size) {
        for (let k = 0; k < half; k++) {
          const wr = this.cos[k * step];
          const wi = this.sin[k * step];
          const a = start + k;
          const b = a + half;
          const xr = re[b] * wr - im[b] * wi;
          const xi = re[b] * wi + im[b] * wr;
          re[b] = re[a] - xr; im[b] = im[a] - xi;
          re[a] += xr; im[a] += xi;
        }
      }
    }
  }
}

export class LogMel {
  /**
   * @param {object} cfg          see DEFAULT_CONFIG
   * @param {object} melFilters   parsed docs/model/mel_filters.json
   */
  constructor(cfg, melFilters) {
    if (melFilters.n_fft !== cfg.nFft || melFilters.n_mels !== cfg.nMels || melFilters.sample_rate !== cfg.sampleRate) {
      throw new Error('mel_filters.json does not match the feature settings');
    }
    this.cfg = cfg;
    this.nSamples = nSamples(cfg);
    this.nFrames = nFrames(cfg);
    this.nBins = cfg.nFft / 2 + 1;
    this.filters = melFilters.filters.map((f) => ({ start: f.start, weights: Float64Array.from(f.weights) }));
    this.fft = new FFT(cfg.nFft);
    // scipy.signal.get_window('hann', n_fft), i.e. the periodic Hann window librosa uses
    this.window = new Float64Array(cfg.nFft);
    for (let i = 0; i < cfg.nFft; i++) this.window[i] = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / cfg.nFft);
    this.re = new Float64Array(cfg.nFft);
    this.im = new Float64Array(cfg.nFft);
    this.power = new Float64Array(this.nBins);
  }

  /** Mel power spectrogram, (nMels, nFrames) row-major, like librosa.feature.melspectrogram. */
  melPower(y) {
    const { nFft, hopLength, nMels } = this.cfg;
    const pad = nFft >> 1;
    const T = 1 + Math.floor(y.length / hopLength);
    const mel = new Float64Array(nMels * T);
    for (let t = 0; t < T; t++) {
      const offset = t * hopLength - pad; // center=True, pad_mode='constant' (zeros)
      for (let j = 0; j < nFft; j++) {
        const k = offset + j;
        this.re[j] = k >= 0 && k < y.length ? y[k] * this.window[j] : 0;
        this.im[j] = 0;
      }
      this.fft.transform(this.re, this.im);
      for (let k = 0; k < this.nBins; k++) this.power[k] = this.re[k] * this.re[k] + this.im[k] * this.im[k];
      for (let m = 0; m < nMels; m++) {
        const { start, weights } = this.filters[m];
        let s = 0;
        for (let i = 0; i < weights.length; i++) s += weights[i] * this.power[start + i];
        mel[m * T + t] = s;
      }
    }
    return { mel, frames: T };
  }

  /**
   * One window of exactly nSamples samples -> Float32Array (nMels * nFrames) in [0, 1],
   * row-major by mel band, which is the model's (128, 173, 1) input layout (features.log_mel).
   */
  compute(y) {
    if (y.length !== this.nSamples) throw new Error(`expected ${this.nSamples} samples, got ${y.length}`);
    const { mel } = this.melPower(y);
    const out = new Float32Array(mel.length);
    let ref = 0;
    for (let i = 0; i < mel.length; i++) if (mel[i] > ref) ref = mel[i];
    if (ref < 1e-10) return out; // digital silence -> zeros
    const amin = 1e-10;
    const { topDb } = this.cfg;
    const refDb = 10 * Math.log10(Math.max(amin, ref));
    // power_to_db(ref=max): the largest value is 0 dB, everything is clipped at -top_db
    for (let i = 0; i < mel.length; i++) {
      const db = Math.max(10 * Math.log10(Math.max(amin, mel[i])) - refDb, -topDb);
      out[i] = (db + topDb) / topDb;
    }
    return out;
  }

  /** Audio of any length -> { features: [Float32Array...], starts: [seconds...] } (features.windows_to_features). */
  windows(y, hopSeconds = 1.0) {
    const n = this.nSamples;
    const hop = Math.max(1, Math.trunc(hopSeconds * this.cfg.sampleRate));
    const starts = windowStarts(y.length, n, hop);
    const features = starts.map((s) => this.compute(loopPad(y.subarray(s, s + n), n)));
    return { features, starts: starts.map((s) => s / this.cfg.sampleRate) };
  }
}
