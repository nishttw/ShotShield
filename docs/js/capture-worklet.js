// AudioWorklet: collects raw microphone samples (mixed to mono) and posts them to the page in chunks.
class CaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.chunk = new Float32Array(2048);
    this.n = 0;
  }

  process(inputs) {
    const channels = inputs[0];
    if (channels && channels.length) {
      const frames = channels[0].length;
      for (let i = 0; i < frames; i++) {
        let s = 0;
        for (let c = 0; c < channels.length; c++) s += channels[c][i];
        this.chunk[this.n++] = s / channels.length;
        if (this.n === this.chunk.length) {
          this.port.postMessage(this.chunk);
          this.chunk = new Float32Array(2048);
          this.n = 0;
        }
      }
    }
    return true;
  }
}

registerProcessor('shotshield-capture', CaptureProcessor);
