// Copyright 2026 Inworld AI & Antigravity
// Real-time Holographic Spectrum & Waveform Visualizers

class MasterVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.analyser = null;
    this.dataArray = null;
    this.isRunning = false;
    this.resize();
    window.addEventListener('resize', () => this.resize());
  }

  resize() {
    if (!this.canvas) return;
    const rect = this.canvas.getBoundingClientRect();
    this.canvas.width = rect.width * window.devicePixelRatio;
    this.canvas.height = rect.height * window.devicePixelRatio;
    this.ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this.width = rect.width;
    this.height = rect.height;
  }

  connect(audioNode) {
    if (!audioNode || !audioNode.context) return;
    const ctx = audioNode.context;
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 256;
    audioNode.connect(this.analyser);
    const bufferLength = this.analyser.frequencyBinCount;
    this.dataArray = new Uint8Array(bufferLength);
  }

  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    this.render();
  }

  stop() {
    this.isRunning = false;
    if (this.ctx) {
      this.ctx.clearRect(0, 0, this.width, this.height);
      this.drawIdle();
    }
  }

  drawIdle() {
    if (!this.ctx) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Draw Logic Pro Channel EQ subtle baseline & grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, this.height / 2);
    ctx.lineTo(this.width, this.height / 2);
    ctx.stroke();

    // Ambient EQ curve
    ctx.strokeStyle = 'rgba(32, 212, 164, 0.4)';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    const time = Date.now() * 0.002;
    for (let x = 0; x < this.width; x += 4) {
      const y = this.height / 2 + Math.sin(x * 0.02 + time) * 6 * Math.sin(x * 0.005);
      if (x === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  render() {
    if (!this.isRunning) return;
    requestAnimationFrame(() => this.render());

    if (!this.analyser || !this.dataArray) {
      this.drawIdle();
      return;
    }

    this.analyser.getByteFrequencyData(this.dataArray);
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.width, this.height);

    // Logic Pro Channel EQ Analyzer Grid
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    for (let gy = 20; gy < this.height; gy += 25) {
      ctx.beginPath();
      ctx.moveTo(0, gy);
      ctx.lineTo(this.width, gy);
      ctx.stroke();
    }

    const barCount = 64;
    const barWidth = (this.width / barCount) * 0.75;
    const gap = (this.width / barCount) * 0.25;

    for (let i = 0; i < barCount; i++) {
      const dataIndex = Math.floor((i / barCount) * this.dataArray.length);
      const val = this.dataArray[dataIndex] / 255.0; // 0 to 1
      const barHeight = Math.max(2, val * (this.height - 8));

      const x = i * (barWidth + gap) + gap / 2;
      const y = this.height - barHeight;

      // Logic Pro Channel EQ signature gradient (Mint/Cyan to Logic Blue)
      const grad = ctx.createLinearGradient(0, y, 0, this.height);
      grad.addColorStop(0, '#20d4a4');
      grad.addColorStop(0.5, '#2888e0');
      grad.addColorStop(1, '#1b3f68');

      ctx.fillStyle = grad;
      ctx.fillRect(x, y, barWidth, barHeight);

      // Peak cap dot
      ctx.fillStyle = '#f0f4f8';
      ctx.fillRect(x, y - 1, barWidth, 1.5);
    }
  }
}

class WaveformRenderer {
  static drawWaveform(canvas, peaks, color = '#2888e0', progress = 0) {
    if (!canvas || !peaks || peaks.length === 0) return;
    const ctx = canvas.getContext('2d');
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * window.devicePixelRatio;
    canvas.height = rect.height * window.devicePixelRatio;
    ctx.scale(window.devicePixelRatio, window.devicePixelRatio);

    const w = rect.width;
    const h = rect.height;
    const midY = h / 2;

    ctx.clearRect(0, 0, w, h);

    const step = w / peaks.length;
    const progressX = progress * w;

    for (let i = 0; i < peaks.length; i++) {
      const x = i * step;
      const [minVal, maxVal] = peaks[i];
      const y1 = midY + minVal * (h * 0.45);
      const y2 = midY + maxVal * (h * 0.45);

      if (x <= progressX) {
        ctx.strokeStyle = color;
      } else {
        ctx.strokeStyle = '#434752';
      }

      ctx.lineWidth = Math.max(1, step * 0.7);
      ctx.beginPath();
      ctx.moveTo(x, Math.min(y1, y2 - 1));
      ctx.lineTo(x, Math.max(y2, y1 + 1));
      ctx.stroke();
    }
  }
}

window.MasterVisualizer = MasterVisualizer;
window.WaveformRenderer = WaveformRenderer;
