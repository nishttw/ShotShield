import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.wasm.min.mjs';
import { LogMel, configFromInfo } from './features.js';
import { StreamingResampler } from './resampler.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1; // GitHub Pages is not cross-origin isolated, so no wasm threads

const $ = (id) => document.getElementById(id);
const HISTORY = 60; // live predictions kept in the chart (one per second)

const S = {
  cfg: null, logmel: null, session: null, inputName: null, outputName: null, threshold: 0.5,
  queue: Promise.resolve(), examples: [],
  live: null, // { ctx, stream, node, resampler, ring, sinceLast, busy, history, event }
};

// ---------- model ----------
function predict(features) {
  // onnxruntime-web runs one inference at a time per session, so calls are queued
  const run = async () => {
    const t0 = performance.now();
    const x = new ort.Tensor('float32', features, [1, S.cfg.nMels, S.logmel.nFrames, 1]);
    const out = await S.session.run({ [S.inputName]: x });
    return { p: out[S.outputName].data[0], ms: performance.now() - t0 };
  };
  const result = S.queue.then(run, run);
  S.queue = result.catch(() => {});
  return result;
}

async function fetchJSON(url) {
  // always revalidate: model_info.json names the model file, so a stale copy would point at an old model
  const r = await fetch(url, { cache: 'no-cache' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

async function init() {
  try {
    const [info, filters] = await Promise.all([fetchJSON('model/model_info.json'), fetchJSON('model/mel_filters.json')]);
    S.cfg = configFromInfo(info.features);
    S.logmel = new LogMel(S.cfg, filters);
    S.threshold = info.threshold;
    $('threshold-label').textContent = pct(S.threshold);
    $('meter-threshold').style.left = pct(S.threshold);
    S.session = await ort.InferenceSession.create(`model/${info.web?.onnx_file ?? 'shotshield_v2.onnx'}`, {
      executionProviders: ['wasm'],
    });
    [S.inputName] = S.session.inputNames;
    [S.outputName] = S.session.outputNames;
    const { ms } = await predict(new Float32Array(S.cfg.nMels * S.logmel.nFrames)); // warm-up
    setStatus(`Model ready. One prediction takes about ${Math.round(ms)} ms on this device.`);
    $('start').disabled = false;
    $('file').disabled = false;
    await loadExamples();
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the model: ${err.message}`, true);
  }
}

// ---------- live microphone ----------
class Ring {
  constructor(n) { this.buf = new Float32Array(n); this.pos = 0; this.filled = 0; }
  push(x) {
    const n = this.buf.length;
    if (x.length >= n) { this.buf.set(x.subarray(x.length - n)); this.pos = 0; this.filled = n; return; }
    const first = Math.min(x.length, n - this.pos);
    this.buf.set(x.subarray(0, first), this.pos);
    this.buf.set(x.subarray(first), 0);
    this.pos = (this.pos + x.length) % n;
    this.filled = Math.min(n, this.filled + x.length);
  }
  latest() { // oldest -> newest
    const out = new Float32Array(this.buf.length);
    out.set(this.buf.subarray(this.pos));
    out.set(this.buf.subarray(0, this.pos), this.buf.length - this.pos);
    return out;
  }
}

async function startLive() {
  $('start').disabled = true;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      // raw audio: noise suppression and auto gain would smear exactly the sharp onsets we look for
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1 },
    });
    const ctx = new AudioContext();
    await ctx.audioWorklet.addModule('js/capture-worklet.js');
    const source = ctx.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(ctx, 'shotshield-capture', { numberOfOutputs: 1 });
    source.connect(node);
    node.connect(ctx.destination); // outputs silence; keeps the node running in every browser
    S.live = {
      ctx, stream, node, resampler: new StreamingResampler(ctx.sampleRate, S.cfg.sampleRate),
      ring: new Ring(S.logmel.nSamples), sinceLast: 0, busy: false, history: [], event: null, level: -100,
    };
    node.port.onmessage = (e) => onAudio(e.data);
    $('stop').disabled = false;
    document.body.classList.add('listening');
    setStatus(`Listening (microphone at ${ctx.sampleRate.toLocaleString()} Hz, resampled to ${S.cfg.sampleRate.toLocaleString()} Hz). Collecting the first 4 seconds…`);
  } catch (err) {
    console.error(err);
    $('start').disabled = false;
    setStatus(err.name === 'NotAllowedError' ? 'Microphone access was blocked. Allow it in the address bar and try again.'
      : `Could not open the microphone: ${err.message}`, true);
  }
}

function stopLive() {
  const L = S.live;
  if (!L) return;
  L.node.port.onmessage = null;
  L.stream.getTracks().forEach((t) => t.stop());
  L.ctx.close();
  S.live = null;
  document.body.classList.remove('listening');
  $('stop').disabled = true;
  $('start').disabled = false;
  $('level').style.width = '0%';
  setStatus('Stopped.');
}

function onAudio(chunk) {
  const L = S.live;
  if (!L) return;
  let peak = 0;
  for (let i = 0; i < chunk.length; i++) peak = Math.max(peak, Math.abs(chunk[i]));
  const db = 20 * Math.log10(peak + 1e-9);
  L.level = Math.max(db, L.level - 3);
  $('level').style.width = `${Math.max(0, Math.min(100, (L.level + 70) / 70 * 100))}%`;

  const y = L.resampler.process(chunk);
  L.ring.push(y);
  L.sinceLast += y.length;
  if (L.ring.filled < L.ring.buf.length) {
    $('fill').textContent = `${(L.ring.filled / S.cfg.sampleRate).toFixed(1)} / 4.0 s`;
    return;
  }
  if (L.sinceLast >= S.cfg.sampleRate && !L.busy) { // one prediction per second of new audio
    L.sinceLast = 0;
    L.busy = true;
    liveStep(L.ring.latest()).finally(() => { L.busy = false; });
  }
}

async function liveStep(window) {
  const t0 = performance.now();
  const feats = S.logmel.compute(window);
  const tFeat = performance.now() - t0;
  const { p, ms } = await predict(feats);
  const L = S.live;
  if (!L) return;
  if (!L.started) {
    L.started = true;
    setStatus(`Listening — microphone at ${L.ctx.sampleRate.toLocaleString()} Hz, resampled to ` +
      `${S.cfg.sampleRate.toLocaleString()} Hz. One prediction per second.`);
  }
  $('fill').textContent = `features ${Math.round(tFeat)} ms · model ${Math.round(ms)} ms`;
  drawSpectrogram($('spec'), feats);
  showProbability(p);
  L.history.push(p);
  if (L.history.length > HISTORY) L.history.shift();
  drawHistory(L.history);

  const hit = p >= S.threshold;
  if (hit && !L.event) {
    L.event = { time: new Date(), peak: p, windows: 1 };
    logEvent(L.event);
  } else if (hit) {
    L.event.peak = Math.max(L.event.peak, p);
    L.event.windows += 1;
    logEvent(L.event, true);
  } else {
    L.event = null;
  }
  setAlert(hit, p);
}

function setAlert(on, p) {
  const el = $('alert');
  el.classList.toggle('on', on);
  el.querySelector('strong').textContent = on ? 'Gunshot detected' : 'No gunshot';
  el.querySelector('span').textContent = on ? `${pct(p)} in the last 4 s` : 'in the last 4 s';
}

function logEvent(ev, update = false) {
  const list = $('events');
  $('events-empty').hidden = true;
  const text = `${ev.time.toLocaleTimeString()} — peak ${pct(ev.peak)} over ${ev.windows} window${ev.windows > 1 ? 's' : ''}`;
  if (update && list.firstElementChild) list.firstElementChild.textContent = text;
  else {
    const li = document.createElement('li');
    li.textContent = text;
    list.prepend(li);
  }
}

// ---------- clips (examples and uploads) ----------
async function decode(arrayBuffer) {
  // decodeAudioData resamples to the context's rate; the example WAVs are already 22,050 Hz
  const ctx = new OfflineAudioContext(1, 1, S.cfg.sampleRate);
  const buf = await ctx.decodeAudioData(arrayBuffer);
  const y = new Float32Array(buf.length);
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const ch = buf.getChannelData(c);
    for (let i = 0; i < y.length; i++) y[i] += ch[i] / buf.numberOfChannels;
  }
  return y;
}

async function analyze(y) {
  const { features, starts } = S.logmel.windows(y, 1.0);
  const probs = [];
  for (const f of features) probs.push((await predict(f)).p);
  return { features, starts, probs };
}

async function loadExamples() {
  let data;
  try {
    data = await fetchJSON('examples/examples.json');
  } catch {
    $('examples-section').hidden = true;
    return;
  }
  S.examples = data.examples;
  const grid = $('examples');
  for (const ex of S.examples) {
    const b = document.createElement('button');
    b.className = 'example' + (ex.is_gunshot ? ' gun' : '');
    b.innerHTML = `<span class="tag">${ex.is_gunshot ? 'gunshot' : 'not a gunshot'}</span>
      <b>${label(ex.class)}</b><small>${ex.duration.toFixed(1)} s${ex.dataset === 'fsd50k' ? ' · FSD50K'
        : ex.note ? ' · hardest negative' : ''}</small>`;
    b.title = ex.note ?? `UrbanSound8K ${ex.slice_file_name} (test fold ${ex.fold})`;
    b.onclick = () => runClip(`${capitalize(label(ex.class))} — ${ex.slice_file_name}`, ex.file, ex);
    grid.append(b);
  }
  $('selftest').disabled = false;
}

async function runClip(title, url, ex = null) {
  const card = $('clip-result');
  card.hidden = false;
  $('clip-title').textContent = title;
  $('clip-verdict').textContent = 'Analysing…';
  $('clip-verdict').className = 'verdict';
  $('clip-windows').replaceChildren();
  $('clip-ref').textContent = '';
  try {
    const bytes = await (await fetch(url)).arrayBuffer();
    $('clip-audio').src = url;
    $('clip-audio').play().catch(() => {});
    const y = await decode(bytes);
    const { features, starts, probs } = await analyze(y);
    const best = probs.indexOf(Math.max(...probs));
    drawSpectrogram($('clip-spec'), features[best]);
    const max = probs[best];
    const hit = max >= S.threshold;
    $('clip-verdict').textContent = `${hit ? 'Gunshot detected' : 'No gunshot'} — highest probability ${pct(max)}` +
      (probs.length > 1 ? ` (window at ${starts[best].toFixed(0)}–${(starts[best] + 4).toFixed(0)} s)` : '');
    $('clip-verdict').className = 'verdict ' + (hit ? 'hit' : 'clear');
    for (const [i, p] of probs.entries()) {
      const bar = document.createElement('div');
      bar.className = 'wbar' + (p >= S.threshold ? ' hit' : '');
      bar.style.setProperty('--p', p);
      bar.title = `${starts[i].toFixed(1)}–${(starts[i] + 4).toFixed(1)} s: ${pct(p)}`;
      $('clip-windows').append(bar);
    }
    if (ex) {
      const diff = Math.max(...probs.map((p, i) => Math.abs(p - ex.keras_window_probs[i])));
      $('clip-ref').textContent = `Ground truth: ${label(ex.class)}. The Python/Keras model gave ${pct(ex.keras_max_prob)} ` +
        `for this clip; this browser got ${pct(max)} (largest difference ${diff.toExponential(1)}).`;
      if (ex.credit) {
        const c = ex.credit;
        const credit = document.createElement('span');
        credit.append(` Sound: “${c.title ?? ex.slice_file_name}” by ${c.uploader ?? 'unknown'} (`);
        const a = document.createElement('a');
        a.href = c.license ?? '#';
        a.textContent = 'license';
        credit.append(a, ', ');
        const src = document.createElement('a');
        src.href = ex.source;
        src.textContent = 'freesound.org';
        credit.append(src, ').');
        $('clip-ref').append(credit);
      }
    }
  } catch (err) {
    console.error(err);
    $('clip-verdict').textContent = `Could not analyse this file: ${err.message}`;
  }
}

async function selfTest() {
  const btn = $('selftest');
  btn.disabled = true;
  const rows = [];
  let worst = 0;
  for (const ex of S.examples) {
    const y = await decode(await (await fetch(ex.file)).arrayBuffer());
    const { probs } = await analyze(y);
    const diff = Math.max(...probs.map((p, i) => Math.abs(p - ex.keras_window_probs[i])));
    worst = Math.max(worst, diff);
    rows.push(`<tr><td>${label(ex.class)}</td><td>${ex.slice_file_name}</td><td>${pct(ex.keras_max_prob, 2)}</td>` +
      `<td>${pct(Math.max(...probs), 2)}</td><td>${diff.toExponential(1)}</td></tr>`);
  }
  $('selftest-out').innerHTML = `<table><thead><tr><th>Clip</th><th>File</th><th>Python (Keras)</th><th>Browser</th>` +
    `<th>Max |diff|</th></tr></thead><tbody>${rows.join('')}</tbody></table>` +
    `<p>${worst < 1e-3 ? '✅' : '⚠️'} Largest difference over all windows: <b>${worst.toExponential(2)}</b> ` +
    `(probabilities are 0–1).</p>`;
  btn.disabled = false;
}

// ---------- drawing ----------
// matplotlib "magma", 9 stops
const MAGMA = [[0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129], [181, 54, 122], [229, 80, 100],
  [251, 135, 97], [254, 194, 135], [252, 253, 191]];
const LUT = Array.from({ length: 256 }, (_, i) => {
  const x = (i / 255) * (MAGMA.length - 1);
  const k = Math.min(MAGMA.length - 2, Math.floor(x));
  const f = x - k;
  return MAGMA[k].map((v, c) => Math.round(v + (MAGMA[k + 1][c] - v) * f));
});

function drawSpectrogram(canvas, feats) {
  const M = S.cfg.nMels;
  const T = S.logmel.nFrames;
  canvas.width = T;
  canvas.height = M;
  const g = canvas.getContext('2d');
  const img = g.createImageData(T, M);
  for (let m = 0; m < M; m++) {
    for (let t = 0; t < T; t++) {
      const [r, gr, b] = LUT[Math.max(0, Math.min(255, Math.round(feats[m * T + t] * 255)))];
      const o = ((M - 1 - m) * T + t) * 4; // low frequencies at the bottom
      img.data[o] = r; img.data[o + 1] = gr; img.data[o + 2] = b; img.data[o + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}

function drawHistory(history) {
  const c = $('history');
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth;
  const h = c.clientHeight;
  c.width = Math.round(w * dpr);
  c.height = Math.round(h * dpr);
  const g = c.getContext('2d');
  g.scale(dpr, dpr);
  const css = getComputedStyle(document.documentElement);
  const bw = w / HISTORY;
  history.forEach((p, i) => {
    const x = w - (history.length - i) * bw;
    g.fillStyle = p >= S.threshold ? css.getPropertyValue('--alert') : css.getPropertyValue('--bar');
    g.fillRect(x + 1, h - p * h, Math.max(1, bw - 2), p * h);
  });
  const ty = h - S.threshold * h;
  g.strokeStyle = css.getPropertyValue('--muted');
  g.setLineDash([4, 4]);
  g.beginPath(); g.moveTo(0, ty); g.lineTo(w, ty); g.stroke();
}

function showProbability(p) {
  $('meter-fill').style.width = pct(p);
  $('meter-fill').classList.toggle('hit', p >= S.threshold);
  $('prob').textContent = pct(p);
}

// ---------- helpers ----------
function pct(x, digits = 0) { return `${(100 * x).toFixed(digits)}%`; }
function label(cls) { return cls.replace(/_/g, ' '); }
function capitalize(s) { return s.charAt(0).toUpperCase() + s.slice(1); }
function setStatus(text, error = false) {
  $('status').textContent = text;
  $('status').classList.toggle('error', error);
}

$('start').onclick = startLive;
$('stop').onclick = stopLive;
$('selftest').onclick = selfTest;
$('file').onchange = (e) => {
  const f = e.target.files[0];
  if (f) runClip(f.name, URL.createObjectURL(f));
};
if (!window.isSecureContext || !navigator.mediaDevices) {
  $('start').title = 'The microphone needs HTTPS';
}
init();
