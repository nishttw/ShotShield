// Streaming sample-rate converter for microphone audio (e.g. 48,000 Hz -> 22,050 Hz).
//
// Browsers resample AudioBuffers with linear interpolation, which folds everything above 11 kHz
// back into the spectrum the model sees. This uses a proper low-pass filter instead: a Kaiser-windowed
// sinc kernel (cut-off just below the output Nyquist frequency), evaluated at fractional positions
// from a precomputed table. librosa.load uses soxr for the same job; both are near-ideal low-pass resamplers.

function besselI0(x) {
  let sum = 1;
  let term = 1;
  for (let k = 1; k < 50; k++) {
    term *= (x / (2 * k)) ** 2;
    sum += term;
    if (term < 1e-12 * sum) break;
  }
  return sum;
}

export class StreamingResampler {
  constructor(inRate, outRate, { zeros = 24, beta = 8.6, rolloff = 0.94, tableRes = 256 } = {}) {
    this.inRate = inRate;
    this.outRate = outRate;
    this.step = inRate / outRate; // input samples per output sample
    this.passthrough = inRate === outRate;
    const fc = 0.5 * Math.min(1, outRate / inRate) * rolloff; // cut-off, cycles per input sample
    this.half = Math.ceil(zeros / (2 * fc)); // kernel half-width in input samples
    this.res = tableRes;
    this.table = new Float32Array(this.half * tableRes + 2);
    const i0b = besselI0(beta);
    for (let i = 0; i < this.table.length; i++) {
      const d = i / tableRes;
      const x = Math.PI * 2 * fc * d;
      const sinc = d === 0 ? 1 : Math.sin(x) / x;
      const r = d / this.half;
      const w = r >= 1 ? 0 : besselI0(beta * Math.sqrt(1 - r * r)) / i0b;
      this.table[i] = 2 * fc * sinc * w;
    }
    this.reset();
  }

  reset() {
    this.buf = new Float32Array(this.half); // zero history before the first sample
    this.pos = this.half; // position of the next output sample, in input samples
  }

  /** Feed input samples, get back every output sample that can be computed so far. */
  process(chunk) {
    if (this.passthrough) return Float32Array.from(chunk);
    const merged = new Float32Array(this.buf.length + chunk.length);
    merged.set(this.buf);
    merged.set(chunk, this.buf.length);
    const buf = merged;
    const { half, res, table, step } = this;
    const last = table.length - 1;
    const out = new Float32Array(Math.ceil((buf.length - this.pos) / step) + 1);
    let n = 0;
    let pos = this.pos;
    while (Math.floor(pos) + half < buf.length) {
      const c = Math.floor(pos);
      let s = 0;
      for (let k = c - half + 1; k <= c + half; k++) {
        const d = Math.abs(k - pos) * res;
        const i = d | 0;
        if (i < last) s += buf[k] * (table[i] + (table[i + 1] - table[i]) * (d - i));
      }
      out[n++] = s;
      pos += step;
    }
    const drop = Math.floor(pos) - half;
    this.buf = drop > 0 ? buf.slice(drop) : buf;
    this.pos = drop > 0 ? pos - drop : pos;
    return out.subarray(0, n);
  }
}
