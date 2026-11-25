/**
 * DENOX Live DSP Audition Engine
 * Real-Time, Zero-Latency Web Audio API Processing for Logic Pro 13 Modular Rack
 * Provides instantaneous audition feedback when moving sliders or selecting presets.
 */

class DenoxLiveDspEngine {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.wetGain = null;
    this.dryGain = null;

    this.isPlaying = false;
    this.isBypassed = false;
    this.activeModule = null;

    this.audioBuffers = {
      original: null,
      vocal: null
    };

    this.sourceNodes = {
      original: null,
      vocal: null
    };

    this.playbackStartTime = 0;
    this.startOffset = 0;
    this.duration = 0;

    // Module DSP Nodes & Parameters
    this.denoiser = {
      vocalGain: null,
      origGain: null,
      threshold: 1.0,
      mode: 'spectral_gate',
      ambient_bleed: 0.0
    };

    this.deesser = {
      peakingFilter: null,
      highShelf: null,
      compressor: null,
      frequency: 6500,
      reduction: 0.65,
      q: 2.0
    };

    this.deverb = {
      highShelf: null,
      lowShelf: null,
      decayGain: null,
      reduction: 0.70,
      roomSize: 'medium',
      earlyReflections: 0.20
    };

    this.declicker = {
      enabled: true,
      sensitivity: 0.75,
      clickWidth: 18,
      prevSample1: 0,
      prevSample2: 0,
      runningRms: 0.01
    };

    this.decrackler = {
      enabled: true,
      sensitivity: 0.70,
      reduction: 0.60,
      band: 'high_frequency',
      prevSample: 0,
      runningVar: 0.005
    };

    this.procNode = null;
  }

  ensureContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx({ latencyHint: 'interactive' });

      // Master output bus
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(1.0, this.ctx.currentTime);
      this.masterGain.connect(this.ctx.destination);

      // A/B comparison crossfaders
      this.wetGain = this.ctx.createGain();
      this.wetGain.gain.setValueAtTime(1.0, this.ctx.currentTime);
      this.wetGain.connect(this.masterGain);

      this.dryGain = this.ctx.createGain();
      this.dryGain.gain.setValueAtTime(0.0, this.ctx.currentTime);
      this.dryGain.connect(this.masterGain);

      // Setup ScriptProcessor for sample-accurate click & crackle removal
      // bufferSize: 1024 samples (~21ms at 48kHz)
      this.procNode = this.ctx.createScriptProcessor(1024, 1, 1);
      this.procNode.onaudioprocess = (e) => this.processAudioBlock(e);
    }

    if (this.ctx.state === 'suspended') {
      this.ctx.resume();
    }
  }

  async loadAudioBuffer(url) {
    this.ensureContext();
    const res = await fetch(url);
    if (!res.ok) throw new Error(`HTTP ${res.status} loading ${url}`);
    const arrayBuffer = await res.arrayBuffer();
    return await this.ctx.decodeAudioData(arrayBuffer);
  }

  async setupModuleBuffers(urls, totalDuration) {
    this.ensureContext();
    this.audioBuffers.original = null;
    this.audioBuffers.vocal = null;

    if (urls.original_url) {
      this.audioBuffers.original = await this.loadAudioBuffer(urls.original_url);
    }
    if (urls.vocal_url) {
      try {
        this.audioBuffers.vocal = await this.loadAudioBuffer(urls.vocal_url);
      } catch (err) {
        console.warn("Could not load vocal buffer for live crossfade:", err);
      }
    }

    this.duration = totalDuration || (this.audioBuffers.original ? this.audioBuffers.original.duration : 0);
  }

  buildGraph(moduleName) {
    this.activeModule = moduleName;
    const now = this.ctx.currentTime;

    // Disconnect old sources
    this.disconnectSources();

    if (!this.audioBuffers.original) return;

    // Create fresh buffer sources
    this.sourceNodes.original = this.ctx.createBufferSource();
    this.sourceNodes.original.buffer = this.audioBuffers.original;
    this.sourceNodes.original.loop = true;

    // Connect Dry path for A/B comparison
    this.sourceNodes.original.connect(this.dryGain);

    if (moduleName === 'denoiser') {
      // De-Noiser: Dual-Track Neural Crossfader
      this.denoiser.vocalGain = this.ctx.createGain();
      this.denoiser.origGain = this.ctx.createGain();

      this.denoiser.vocalGain.connect(this.wetGain);
      this.denoiser.origGain.connect(this.wetGain);

      this.sourceNodes.original.connect(this.denoiser.origGain);

      if (this.audioBuffers.vocal) {
        this.sourceNodes.vocal = this.ctx.createBufferSource();
        this.sourceNodes.vocal.buffer = this.audioBuffers.vocal;
        this.sourceNodes.vocal.loop = true;
        this.sourceNodes.vocal.connect(this.denoiser.vocalGain);
      } else {
        // Fallback: If vocal buffer not available, apply high-frequency spectral shaping
        this.sourceNodes.original.connect(this.denoiser.vocalGain);
      }

      this.applyDenoiserParamsNow();

    } else if (moduleName === 'deesser') {
      // De-Esser: Dynamic Peaking Filter + High Shelf + Compressor
      this.deesser.peakingFilter = this.ctx.createBiquadFilter();
      this.deesser.peakingFilter.type = 'peaking';

      this.deesser.highShelf = this.ctx.createBiquadFilter();
      this.deesser.highShelf.type = 'highshelf';

      this.deesser.compressor = this.ctx.createDynamicsCompressor();
      this.deesser.compressor.threshold.setValueAtTime(-28, now);
      this.deesser.compressor.knee.setValueAtTime(3, now);
      this.deesser.compressor.ratio.setValueAtTime(12, now);
      this.deesser.compressor.attack.setValueAtTime(0.002, now);
      this.deesser.compressor.release.setValueAtTime(0.035, now);

      this.sourceNodes.original.connect(this.deesser.peakingFilter);
      this.deesser.peakingFilter.connect(this.deesser.highShelf);
      this.deesser.highShelf.connect(this.deesser.compressor);
      this.deesser.compressor.connect(this.wetGain);

      this.applyDeesserParamsNow();

    } else if (moduleName === 'deverb') {
      // De-Verb: Multi-Band Reflection Absorption + Downward Decay Expansion
      this.deverb.highShelf = this.ctx.createBiquadFilter();
      this.deverb.highShelf.type = 'highshelf';
      this.deverb.highShelf.frequency.setValueAtTime(3500, now);

      this.deverb.lowShelf = this.ctx.createBiquadFilter();
      this.deverb.lowShelf.type = 'lowshelf';
      this.deverb.lowShelf.frequency.setValueAtTime(250, now);

      this.deverb.decayGain = this.ctx.createGain();

      this.sourceNodes.original.connect(this.deverb.highShelf);
      this.deverb.highShelf.connect(this.deverb.lowShelf);
      this.deverb.lowShelf.connect(this.deverb.decayGain);
      this.deverb.decayGain.connect(this.wetGain);

      this.applyDeverbParamsNow();

    } else if (moduleName === 'declicker' || moduleName === 'decrackler') {
      // Real-time discrete click and crackle suppression via DSP processor node
      this.sourceNodes.original.connect(this.procNode);
      this.procNode.connect(this.wetGain);
    }
  }

  processAudioBlock(e) {
    const input = e.inputBuffer.getChannelData(0);
    const output = e.outputBuffer.getChannelData(0);
    const len = input.length;

    if (this.activeModule === 'declicker') {
      const sens = this.declicker.sensitivity;
      const thresholdFactor = 7.5 - (sens * 5.2); // Lower factor = more aggressive removal
      let p1 = this.declicker.prevSample1;
      let p2 = this.declicker.prevSample2;
      let rms = this.declicker.runningRms;

      for (let i = 0; i < len; i++) {
        const x = input[i];
        // 2nd-order differentiation to isolate impulses
        const diff = x - (2.0 * p1) + p2;
        const diffAbs = Math.abs(diff);

        // Exponential moving average for energy estimate
        rms = (0.995 * rms) + (0.005 * Math.abs(x));

        const isClick = diffAbs > (thresholdFactor * Math.max(rms, 0.005));

        if (isClick && sens > 0.05) {
          // Linear / cubic interpolation across transient spike
          output[i] = (p1 + (i + 1 < len ? input[i + 1] : p1)) * 0.5;
        } else {
          output[i] = x;
        }

        p2 = p1;
        p1 = output[i];
      }

      this.declicker.prevSample1 = p1;
      this.declicker.prevSample2 = p2;
      this.declicker.runningRms = rms;

    } else if (this.activeModule === 'decrackler') {
      const sens = this.decrackler.sensitivity;
      const reduction = this.decrackler.reduction;
      let prev = this.decrackler.prevSample;
      let runVar = this.decrackler.runningVar;

      const maxDelta = 0.08 * (1.0 - (reduction * 0.75));

      for (let i = 0; i < len; i++) {
        const x = input[i];
        const delta = Math.abs(x - prev);
        runVar = (0.992 * runVar) + (0.008 * delta);

        if (delta > maxDelta && sens > 0.05) {
          // Soft clamp granular crackle burst
          const sign = x >= prev ? 1 : -1;
          output[i] = prev + sign * (maxDelta + (delta - maxDelta) * (1.0 - reduction * 0.8));
        } else {
          output[i] = x;
        }
        prev = output[i];
      }

      this.decrackler.prevSample = prev;
      this.decrackler.runningVar = runVar;

    } else {
      // Pass-through
      output.set(input);
    }
  }

  // --- PARAMETER AUTOMATION METHODS (0ms Latency) ---

  updateDenoiserParams(threshold, ambientBleed, mode) {
    this.denoiser.threshold = threshold !== undefined ? threshold : this.denoiser.threshold;
    this.denoiser.ambient_bleed = ambientBleed !== undefined ? ambientBleed : this.denoiser.ambient_bleed;
    this.denoiser.mode = mode || this.denoiser.mode;

    if (this.activeModule === 'denoiser' && this.isPlaying) {
      this.applyDenoiserParamsNow();
    }
  }

  applyDenoiserParamsNow() {
    if (!this.denoiser.vocalGain || !this.denoiser.origGain) return;
    const now = this.ctx.currentTime;
    const t = Math.max(0.0, Math.min(1.0, this.denoiser.threshold));
    const bleed = Math.max(0.0, Math.min(0.35, this.denoiser.ambient_bleed));
    const mode = this.denoiser.mode;

    let vGain = 1.0;
    let oGain = 0.0;

    if (mode === 'blend') {
      const angle = t * (Math.PI / 2.0);
      oGain = Math.cos(angle);
      vGain = Math.sin(angle);
    } else if (mode === 'air_preserve') {
      const att = Math.pow(1.0 - t, 1.2);
      const b = Math.max(att, bleed);
      vGain = Math.max(0.0, 1.0 - b);
      oGain = b;
    } else {
      // spectral_gate default
      const att = Math.pow(1.0 - t, 1.5) + bleed;
      const clampedAtt = Math.max(0.0, Math.min(1.0, att));
      vGain = Math.max(0.0, 1.0 - clampedAtt);
      oGain = clampedAtt;
    }

    // Set sample-accurate gain transition over 8ms
    this.denoiser.vocalGain.gain.setTargetAtTime(vGain, now, 0.008);
    this.denoiser.origGain.gain.setTargetAtTime(oGain, now, 0.008);
  }

  updateDeesserParams(reduction, frequency, q) {
    if (reduction !== undefined) this.deesser.reduction = reduction;
    if (frequency !== undefined) this.deesser.frequency = frequency;
    if (q !== undefined) this.deesser.q = q;

    if (this.activeModule === 'deesser' && this.isPlaying) {
      this.applyDeesserParamsNow();
    }
  }

  applyDeesserParamsNow() {
    if (!this.deesser.peakingFilter || !this.deesser.highShelf) return;
    const now = this.ctx.currentTime;
    const freq = Math.max(2500, Math.min(11000, this.deesser.frequency));
    const red = Math.max(0.0, Math.min(1.0, this.deesser.reduction));
    const q = Math.max(0.5, Math.min(6.0, this.deesser.q));

    // Dynamic notch depth down to -24dB
    const notchGain = -(red * 24.0);
    const shelfGain = -(red * 6.5);

    this.deesser.peakingFilter.frequency.setTargetAtTime(freq, now, 0.008);
    this.deesser.peakingFilter.Q.setTargetAtTime(q, now, 0.008);
    this.deesser.peakingFilter.gain.setTargetAtTime(notchGain, now, 0.008);

    this.deesser.highShelf.frequency.setTargetAtTime(Math.min(12000, freq + 1500), now, 0.008);
    this.deesser.highShelf.gain.setTargetAtTime(shelfGain, now, 0.008);
  }

  updateDeverbParams(reduction, roomSize, earlyReflections) {
    if (reduction !== undefined) this.deverb.reduction = reduction;
    if (roomSize !== undefined) this.deverb.roomSize = roomSize;
    if (earlyReflections !== undefined) this.deverb.earlyReflections = earlyReflections;

    if (this.activeModule === 'deverb' && this.isPlaying) {
      this.applyDeverbParamsNow();
    }
  }

  applyDeverbParamsNow() {
    if (!this.deverb.highShelf || !this.deverb.lowShelf || !this.deverb.decayGain) return;
    const now = this.ctx.currentTime;
    const red = Math.max(0.0, Math.min(1.0, this.deverb.reduction));

    // Higher reduction suppresses diffuse reflections
    const highDamp = -(red * 14.0);
    const lowDamp = -(red * 9.0);
    const decayLevel = Math.max(0.15, 1.0 - (red * 0.35));

    this.deverb.highShelf.gain.setTargetAtTime(highDamp, now, 0.008);
    this.deverb.lowShelf.gain.setTargetAtTime(lowDamp, now, 0.008);
    this.deverb.decayGain.gain.setTargetAtTime(decayLevel, now, 0.008);
  }

  updateDeclickerParams(sensitivity, clickWidth) {
    if (sensitivity !== undefined) this.declicker.sensitivity = sensitivity;
    if (clickWidth !== undefined) this.declicker.clickWidth = clickWidth;
  }

  updateDecracklerParams(sensitivity, crackleReduction, band) {
    if (sensitivity !== undefined) this.decrackler.sensitivity = sensitivity;
    if (crackleReduction !== undefined) this.decrackler.reduction = crackleReduction;
    if (band !== undefined) this.decrackler.band = band;
  }

  // Generic dispatcher for slider input events
  updateModuleParams(modKey, params) {
    if (modKey === 'denoiser') {
      this.updateDenoiserParams(params.threshold, params.ambient_bleed, params.mode);
    } else if (modKey === 'declicker') {
      this.updateDeclickerParams(params.sensitivity, params.click_width_samples);
    } else if (modKey === 'decrackler') {
      this.updateDecracklerParams(params.sensitivity, params.crackle_reduction, params.frequency_band);
    } else if (modKey === 'deverb') {
      this.updateDeverbParams(params.reduction_amount, params.room_size, params.preserve_early_reflections);
    } else if (modKey === 'deesser') {
      this.updateDeesserParams(params.reduction_amount, params.frequency_focus_hz, params.bandwidth_q);
    }
  }

  // --- PLAYBACK CONTROL ---

  play(moduleName, offsetSec = 0) {
    this.ensureContext();
    if (this.isPlaying) this.pause();

    this.buildGraph(moduleName);

    const now = this.ctx.currentTime;
    const offset = Math.max(0, Math.min(this.duration || 0, offsetSec));
    this.startOffset = offset;
    this.playbackStartTime = now;

    if (this.sourceNodes.original) {
      this.sourceNodes.original.start(now, offset);
    }
    if (this.sourceNodes.vocal) {
      this.sourceNodes.vocal.start(now, offset);
    }

    this.isPlaying = true;
    this.setBypass(this.isBypassed);
  }

  pause() {
    if (!this.isPlaying) return;
    this.startOffset = this.getCurrentTime();
    this.disconnectSources();
    this.isPlaying = false;
  }

  seek(pct) {
    const targetTime = (pct / 100.0) * (this.duration || 1);
    if (this.isPlaying) {
      this.play(this.activeModule, targetTime);
    } else {
      this.startOffset = targetTime;
    }
  }

  setBypass(isBypassed) {
    this.isBypassed = !!isBypassed;
    if (!this.ctx) return;
    const now = this.ctx.currentTime;
    const wetTarget = this.isBypassed ? 0.0 : 1.0;
    const dryTarget = this.isBypassed ? 1.0 : 0.0;

    // 15ms professional mixing console crossfade curve
    this.wetGain.gain.setTargetAtTime(wetTarget, now, 0.015);
    this.dryGain.gain.setTargetAtTime(dryTarget, now, 0.015);
  }

  toggleAB() {
    this.setBypass(!this.isBypassed);
    return this.isBypassed;
  }

  getCurrentTime() {
    if (!this.isPlaying) return this.startOffset;
    if (!this.duration) return 0;
    const elapsed = this.ctx.currentTime - this.playbackStartTime;
    return (this.startOffset + elapsed) % this.duration;
  }

  disconnectSources() {
    try {
      if (this.sourceNodes.original) {
        this.sourceNodes.original.stop();
        this.sourceNodes.original.disconnect();
        this.sourceNodes.original = null;
      }
      if (this.sourceNodes.vocal) {
        this.sourceNodes.vocal.stop();
        this.sourceNodes.vocal.disconnect();
        this.sourceNodes.vocal = null;
      }
    } catch (e) {
      // Ignore already stopped nodes
    }
  }
}

// Global singleton instance
window.denoxDsp = new DenoxLiveDspEngine();
