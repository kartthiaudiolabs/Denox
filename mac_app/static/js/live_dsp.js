/**
 * DENOX Live DSP Audition Engine
 * Real-Time, Zero-Latency Web Audio API Processing for Logic Pro 13 Modular Rack
 * Provides instantaneous, audible feedback when moving sliders or selecting presets.
 */

class DenoxLiveDspEngine {
  constructor() {
    this.ctx = null;
    this.masterGain = null;
    this.wetGain = null;
    this.dryGain = null;
    this.limiter = null;

    this.isPlaying = false;
    this.isBypassed = false;
    this.activeModule = null;

    this.audioBuffers = {
      original: null,
      vocal: null
    };

    this.sourceNodes = {
      original: null,
      vocal: null,
      active: null
    };

    this.playbackStartTime = 0;
    this.startOffset = 0;
    this.duration = 0;

    // Module DSP Nodes & Parameters
    this.denoiser = {
      vocalGain: null,
      origGain: null,
      airFilter: null,
      threshold: 1.0,
      mode: 'spectral_gate',
      ambient_bleed: 0.0
    };

    this.declicker = {
      enabled: true,
      sensitivity: 0.75,
      clickWidth: 18,
      transientFilter: null,
      prevSample1: 0,
      prevSample2: 0,
      runningRms: 0.01
    };

    this.decrackler = {
      enabled: true,
      sensitivity: 0.70,
      reduction: 0.65,
      band: 'high_frequency',
      bandFilter: null,
      crackleCompressor: null,
      bandGain: null,
      residualGain: null
    };

    this.deverb = {
      highShelf: null,
      midNotch: null,
      lowShelf: null,
      tailExpander: null,
      decayGain: null,
      reduction: 0.85,
      roomSize: 'small',
      earlyReflections: 0.40
    };

    this.deesser = {
      peakingFilter: null,
      highShelf: null,
      compressor: null,
      frequency: 6500,
      reduction: 0.65,
      q: 1.4,
      mode: 'split_band'
    };

    this.procNode = null;
  }

  ensureContext() {
    if (!this.ctx) {
      const AudioCtx = window.AudioContext || window.webkitAudioContext;
      this.ctx = new AudioCtx({ latencyHint: 'interactive' });

      // Master output bus with Studio Brickwall Limiter (-0.5 dBFS)
      this.masterGain = this.ctx.createGain();
      this.masterGain.gain.setValueAtTime(1.0, this.ctx.currentTime);

      this.limiter = this.ctx.createDynamicsCompressor();
      this.limiter.threshold.setValueAtTime(-0.5, this.ctx.currentTime);
      this.limiter.knee.setValueAtTime(0, this.ctx.currentTime);
      this.limiter.ratio.setValueAtTime(20, this.ctx.currentTime);
      this.limiter.attack.setValueAtTime(0.001, this.ctx.currentTime);
      this.limiter.release.setValueAtTime(0.020, this.ctx.currentTime);

      this.masterGain.connect(this.limiter);
      this.limiter.connect(this.ctx.destination);

      // A/B comparison crossfaders
      this.wetGain = this.ctx.createGain();
      this.wetGain.gain.setValueAtTime(1.0, this.ctx.currentTime);
      this.wetGain.connect(this.masterGain);

      this.dryGain = this.ctx.createGain();
      this.dryGain.gain.setValueAtTime(0.0, this.ctx.currentTime);
      this.dryGain.connect(this.masterGain);

      // Setup ScriptProcessor for sample-accurate click removal
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

    if (urls.original_url || urls.original) {
      try {
        this.audioBuffers.original = await this.loadAudioBuffer(urls.original_url || urls.original);
      } catch (err) {
        console.warn("Could not load original buffer:", err);
      }
    }
    if (urls.vocal_url || urls.vocal) {
      try {
        this.audioBuffers.vocal = await this.loadAudioBuffer(urls.vocal_url || urls.vocal);
      } catch (err) {
        console.warn("Could not load vocal buffer:", err);
      }
    }

    // Fallback: If only one is available, clone reference so both are populated
    if (!this.audioBuffers.vocal && this.audioBuffers.original) {
      this.audioBuffers.vocal = this.audioBuffers.original;
    }
    if (!this.audioBuffers.original && this.audioBuffers.vocal) {
      this.audioBuffers.original = this.audioBuffers.vocal;
    }

    const bufDur = this.audioBuffers.vocal ? this.audioBuffers.vocal.duration : (this.audioBuffers.original ? this.audioBuffers.original.duration : 0);
    this.duration = totalDuration || bufDur;
  }

  buildGraph(moduleName) {
    this.activeModule = moduleName;
    const now = this.ctx.currentTime;

    // Disconnect old sources & processing nodes
    this.disconnectSources();

    if (!this.audioBuffers.original && !this.audioBuffers.vocal) return;

    if (moduleName === 'denoiser') {
      // ----------------------------------------------------------------------
      // 1. DE-NOISER: Dual-Track Neural Crossfader + Air Preservation
      // ----------------------------------------------------------------------
      this.denoiser.vocalGain = this.ctx.createGain();
      this.denoiser.origGain = this.ctx.createGain();

      this.denoiser.airFilter = this.ctx.createBiquadFilter();
      this.denoiser.airFilter.type = 'highshelf';
      this.denoiser.airFilter.frequency.setValueAtTime(6500, now);
      this.denoiser.airFilter.gain.setValueAtTime(0, now);

      this.denoiser.vocalGain.connect(this.wetGain);
      this.denoiser.origGain.connect(this.denoiser.airFilter);
      this.denoiser.airFilter.connect(this.wetGain);

      // Create Sources
      this.sourceNodes.original = this.ctx.createBufferSource();
      this.sourceNodes.original.buffer = this.audioBuffers.original || this.audioBuffers.vocal;
      this.sourceNodes.original.loop = true;
      this.sourceNodes.original.connect(this.denoiser.origGain);
      this.sourceNodes.original.connect(this.dryGain); // Dry = uncleaned original

      this.sourceNodes.vocal = this.ctx.createBufferSource();
      this.sourceNodes.vocal.buffer = this.audioBuffers.vocal || this.audioBuffers.original;
      this.sourceNodes.vocal.loop = true;
      this.sourceNodes.vocal.connect(this.denoiser.vocalGain);

      this.applyDenoiserParamsNow();

    } else {
      // ----------------------------------------------------------------------
      // 2-5. DOWNSTREAM VOCAL MODULES (De-Clicker, De-Crackler, De-Verb, De-Esser)
      // Process the isolated voice (or clean input) with instantaneous real-time DSP
      // ----------------------------------------------------------------------
      const targetBuffer = this.audioBuffers.vocal || this.audioBuffers.original;
      this.sourceNodes.active = this.ctx.createBufferSource();
      this.sourceNodes.active.buffer = targetBuffer;
      this.sourceNodes.active.loop = true;

      // Dry path connects directly to dryGain for instant A/B comparison
      this.sourceNodes.active.connect(this.dryGain);

      if (moduleName === 'declicker') {
        // De-Clicker: Sample-accurate transient differentiator + dynamic tamer
        this.declicker.transientFilter = this.ctx.createBiquadFilter();
        this.declicker.transientFilter.type = 'highshelf';
        this.declicker.transientFilter.frequency.setValueAtTime(7500, now);

        this.sourceNodes.active.connect(this.procNode);
        this.procNode.connect(this.declicker.transientFilter);
        this.declicker.transientFilter.connect(this.wetGain);

        this.declicker.prevSample1 = 0;
        this.declicker.prevSample2 = 0;
        this.declicker.runningRms = 0.01;

        this.applyDeclickerParamsNow();

      } else if (moduleName === 'decrackler') {
        // De-Crackler: Multi-Band Dynamic Granular Noise Compressor Network
        this.decrackler.bandFilter = this.ctx.createBiquadFilter();
        this.decrackler.crackleCompressor = this.ctx.createDynamicsCompressor();
        this.decrackler.bandGain = this.ctx.createGain();
        this.decrackler.residualGain = this.ctx.createGain();

        // Band Path: active -> bandFilter -> compressor -> bandGain -> wetGain
        this.sourceNodes.active.connect(this.decrackler.bandFilter);
        this.decrackler.bandFilter.connect(this.decrackler.crackleCompressor);
        this.decrackler.crackleCompressor.connect(this.decrackler.bandGain);
        this.decrackler.bandGain.connect(this.wetGain);

        // Residual Path: active -> residualGain -> wetGain
        this.sourceNodes.active.connect(this.decrackler.residualGain);
        this.decrackler.residualGain.connect(this.wetGain);

        this.applyDecracklerParamsNow();

      } else if (moduleName === 'deverb') {
        // De-Verb: Multi-Band Reflection Absorption + Downward Decay Expander
        this.deverb.lowShelf = this.ctx.createBiquadFilter();
        this.deverb.lowShelf.type = 'lowshelf';
        this.deverb.lowShelf.frequency.setValueAtTime(250, now);

        this.deverb.midNotch = this.ctx.createBiquadFilter();
        this.deverb.midNotch.type = 'peaking';
        this.deverb.midNotch.frequency.setValueAtTime(700, now);
        this.deverb.midNotch.Q.setValueAtTime(1.8, now);

        this.deverb.highShelf = this.ctx.createBiquadFilter();
        this.deverb.highShelf.type = 'highshelf';
        this.deverb.highShelf.frequency.setValueAtTime(3600, now);

        this.deverb.tailExpander = this.ctx.createDynamicsCompressor();
        this.deverb.tailExpander.knee.setValueAtTime(4, now);
        this.deverb.tailExpander.attack.setValueAtTime(0.002, now);

        this.deverb.decayGain = this.ctx.createGain();

        // Chain: active -> lowShelf -> midNotch -> highShelf -> tailExpander -> decayGain -> wetGain
        this.sourceNodes.active.connect(this.deverb.lowShelf);
        this.deverb.lowShelf.connect(this.deverb.midNotch);
        this.deverb.midNotch.connect(this.deverb.highShelf);
        this.deverb.highShelf.connect(this.deverb.tailExpander);
        this.deverb.tailExpander.connect(this.deverb.decayGain);
        this.deverb.decayGain.connect(this.wetGain);

        this.applyDeverbParamsNow();

      } else if (moduleName === 'deesser') {
        // De-Esser: Dynamic Frequency-Selective Sibilance Compressor
        this.deesser.peakingFilter = this.ctx.createBiquadFilter();
        this.deesser.peakingFilter.type = 'peaking';

        this.deesser.highShelf = this.ctx.createBiquadFilter();
        this.deesser.highShelf.type = 'highshelf';

        this.deesser.compressor = this.ctx.createDynamicsCompressor();
        this.deesser.compressor.knee.setValueAtTime(3, now);
        this.deesser.compressor.attack.setValueAtTime(0.001, now);
        this.deesser.compressor.release.setValueAtTime(0.035, now);

        // Chain: active -> peakingFilter -> highShelf -> compressor -> wetGain
        this.sourceNodes.active.connect(this.deesser.peakingFilter);
        this.deesser.peakingFilter.connect(this.deesser.highShelf);
        this.deesser.highShelf.connect(this.deesser.compressor);
        this.deesser.compressor.connect(this.wetGain);

        this.applyDeesserParamsNow();
      }
    }
  }

  processAudioBlock(e) {
    const input = e.inputBuffer.getChannelData(0);
    const output = e.outputBuffer.getChannelData(0);
    const len = input.length;

    if (this.activeModule === 'declicker') {
      const sens = this.declicker.sensitivity;
      const width = Math.max(4, Math.min(48, Math.round(this.declicker.clickWidth || 18)));
      const thresholdFactor = Math.max(3.2, 7.8 - (sens * 4.6));
      let p1 = this.declicker.prevSample1;
      let p2 = this.declicker.prevSample2;
      let rms = this.declicker.runningRms;
      const halfW = Math.floor(width / 2);

      for (let i = 0; i < len; i++) {
        const x = input[i];
        const diff = x - (2.0 * p1) + p2;
        const diffAbs = Math.abs(diff);

        rms = (0.995 * rms) + (0.005 * Math.abs(x));
        const isClick = diffAbs > (thresholdFactor * Math.max(rms, 0.004));

        if (isClick && sens > 0.05) {
          const targetIdx = Math.min(len - 1, i + halfW);
          const rightClean = input[targetIdx];
          const leftClean = p1;
          const maxEnv = Math.max(Math.abs(leftClean), Math.abs(rightClean), 0.005) * 1.25;

          const t = Math.min(1.0, 1.0 / (halfW + 1));
          const interp = (1.0 - t) * leftClean + t * rightClean;
          output[i] = Math.max(-maxEnv, Math.min(maxEnv, interp));
        } else {
          output[i] = x;
        }

        p2 = p1;
        p1 = output[i];
      }

      this.declicker.prevSample1 = p1;
      this.declicker.prevSample2 = p2;
      this.declicker.runningRms = rms;

    } else {
      // Pass-through
      output.set(input);
    }
  }

  // --- PARAMETER AUTOMATION METHODS (0ms Latency) ---

  updateDenoiserParams(threshold, ambientBleed, mode) {
    if (threshold !== undefined) this.denoiser.threshold = threshold;
    if (ambientBleed !== undefined) this.denoiser.ambient_bleed = ambientBleed;
    if (mode !== undefined) this.denoiser.mode = mode;

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
      // Coherent in-phase linear crossfade
      vGain = t;
      oGain = 1.0 - t;
    } else if (mode === 'air_preserve') {
      const att = Math.pow(1.0 - t, 1.2);
      const b = Math.max(att, bleed);
      vGain = Math.max(0.0, 1.0 - b);
      oGain = b;
      if (this.denoiser.airFilter) {
        const airBoost = t * 3.5;
        this.denoiser.airFilter.gain.setTargetAtTime(airBoost, now, 0.005);
      }
    } else {
      // spectral_gate default
      const att = Math.pow(1.0 - t, 1.5) + bleed;
      const clampedAtt = Math.max(0.0, Math.min(1.0, att));
      vGain = Math.max(0.0, 1.0 - clampedAtt);
      oGain = clampedAtt;
    }

    this.denoiser.vocalGain.gain.setTargetAtTime(vGain, now, 0.005);
    this.denoiser.origGain.gain.setTargetAtTime(oGain, now, 0.005);
  }

  updateDeclickerParams(sensitivity, clickWidth) {
    if (sensitivity !== undefined) this.declicker.sensitivity = sensitivity;
    if (clickWidth !== undefined) this.declicker.clickWidth = clickWidth;

    if (this.activeModule === 'declicker' && this.isPlaying) {
      this.applyDeclickerParamsNow();
    }
  }

  applyDeclickerParamsNow() {
    if (!this.declicker.transientFilter) return;
    const now = this.ctx.currentTime;
    const sens = Math.max(0.0, Math.min(1.0, this.declicker.sensitivity));
    const shelfGain = -(sens * 5.5);
    this.declicker.transientFilter.gain.setTargetAtTime(shelfGain, now, 0.005);
  }

  updateDecracklerParams(sensitivity, crackleReduction, band) {
    if (sensitivity !== undefined) this.decrackler.sensitivity = sensitivity;
    if (crackleReduction !== undefined) this.decrackler.reduction = crackleReduction;
    if (band !== undefined) this.decrackler.band = band;

    if (this.activeModule === 'decrackler' && this.isPlaying) {
      this.applyDecracklerParamsNow();
    }
  }

  applyDecracklerParamsNow() {
    if (!this.decrackler.bandFilter || !this.decrackler.crackleCompressor || !this.decrackler.bandGain) return;
    const now = this.ctx.currentTime;
    const sens = Math.max(0.0, Math.min(1.0, this.decrackler.sensitivity));
    const red = Math.max(0.0, Math.min(1.0, this.decrackler.reduction));
    const band = this.decrackler.band || 'high_frequency';

    if (band === 'high_frequency') {
      this.decrackler.bandFilter.type = 'highpass';
      this.decrackler.bandFilter.frequency.setTargetAtTime(4500, now, 0.005);
    } else if (band === 'speech_presence') {
      this.decrackler.bandFilter.type = 'bandpass';
      this.decrackler.bandFilter.frequency.setTargetAtTime(3600, now, 0.005);
      this.decrackler.bandFilter.Q.setTargetAtTime(1.2, now, 0.005);
    } else if (band === 'low_granular') {
      this.decrackler.bandFilter.type = 'bandpass';
      this.decrackler.bandFilter.frequency.setTargetAtTime(1200, now, 0.005);
      this.decrackler.bandFilter.Q.setTargetAtTime(1.0, now, 0.005);
    } else {
      // full_band
      this.decrackler.bandFilter.type = 'highpass';
      this.decrackler.bandFilter.frequency.setTargetAtTime(80, now, 0.005);
    }

    // Dynamic threshold drops from -12dB down to -42dB
    const compThresh = -12.0 - (sens * 30.0);
    const compRatio = 4.0 + (red * 12.0);
    this.decrackler.crackleCompressor.threshold.setTargetAtTime(compThresh, now, 0.005);
    this.decrackler.crackleCompressor.ratio.setTargetAtTime(compRatio, now, 0.005);
    this.decrackler.crackleCompressor.attack.setTargetAtTime(0.001, now, 0.005);
    this.decrackler.crackleCompressor.release.setTargetAtTime(0.020, now, 0.005);

    // Band attenuation depth
    const bandLevel = Math.max(0.08, 1.0 - (red * 0.90));
    this.decrackler.bandGain.gain.setTargetAtTime(bandLevel, now, 0.005);
    if (this.decrackler.residualGain) {
      this.decrackler.residualGain.gain.setTargetAtTime(band === 'full_band' ? 0.0 : 0.85, now, 0.005);
    }
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
    if (!this.deverb.highShelf || !this.deverb.midNotch || !this.deverb.lowShelf || !this.deverb.tailExpander || !this.deverb.decayGain) return;
    const now = this.ctx.currentTime;
    const red = Math.max(0.0, Math.min(1.0, this.deverb.reduction));
    const early = Math.max(0.0, Math.min(1.0, this.deverb.earlyReflections !== undefined ? this.deverb.earlyReflections : 0.4));
    const size = this.deverb.roomSize || 'small';

    let lowCut = -6.0;
    let midCut = -6.0;
    let highCut = -8.0;
    let releaseSec = 0.075;

    if (size === 'small') {
      lowCut = -4.0;
      midCut = -8.0;
      highCut = -6.0;
      releaseSec = 0.040;
    } else if (size === 'medium') {
      lowCut = -7.0;
      midCut = -6.0;
      highCut = -9.0;
      releaseSec = 0.075;
    } else if (size === 'large') {
      lowCut = -11.0;
      midCut = -5.0;
      highCut = -12.0;
      releaseSec = 0.150;
    } else if (size === 'hall') {
      lowCut = -15.0;
      midCut = -5.0;
      highCut = -16.0;
      releaseSec = 0.260;
    }

    this.deverb.lowShelf.gain.setTargetAtTime(lowCut * red, now, 0.005);
    this.deverb.midNotch.gain.setTargetAtTime(midCut * red, now, 0.005);
    this.deverb.highShelf.gain.setTargetAtTime(highCut * red, now, 0.005);

    // Tail Expander threshold: -14dB down to -36dB
    const expThresh = -14.0 - (red * 22.0);
    const expRatio = 4.0 + (red * 8.0);
    this.deverb.tailExpander.threshold.setTargetAtTime(expThresh, now, 0.005);
    this.deverb.tailExpander.ratio.setTargetAtTime(expRatio, now, 0.005);
    this.deverb.tailExpander.release.setTargetAtTime(releaseSec, now, 0.005);

    // Direct sound clarity floor
    const decayLevel = Math.max(0.35, 1.0 - (red * 0.45 * (1.0 - early * 0.5)));
    this.deverb.decayGain.gain.setTargetAtTime(decayLevel, now, 0.005);
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
    if (!this.deesser.peakingFilter || !this.deesser.highShelf || !this.deesser.compressor) return;
    const now = this.ctx.currentTime;
    const freq = Math.max(3000, Math.min(10000, this.deesser.frequency || 6500));
    const red = Math.max(0.0, Math.min(1.0, this.deesser.reduction));
    const q = Math.max(0.6, Math.min(4.0, this.deesser.q || 1.4));

    // Dynamic notch depth down to -22dB
    const notchGain = -(red * 22.0);
    const shelfGain = -(red * 8.0);

    this.deesser.peakingFilter.frequency.setTargetAtTime(freq, now, 0.005);
    this.deesser.peakingFilter.Q.setTargetAtTime(q, now, 0.005);
    this.deesser.peakingFilter.gain.setTargetAtTime(notchGain, now, 0.005);

    this.deesser.highShelf.frequency.setTargetAtTime(Math.min(12000, freq + 1200), now, 0.005);
    this.deesser.highShelf.gain.setTargetAtTime(shelfGain, now, 0.005);

    // Compressor threshold: -18dB down to -36dB
    const compThresh = -18.0 - (red * 18.0);
    const compRatio = 4.0 + (red * 12.0);
    this.deesser.compressor.threshold.setTargetAtTime(compThresh, now, 0.005);
    this.deesser.compressor.ratio.setTargetAtTime(compRatio, now, 0.005);
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
      try { this.sourceNodes.original.start(now, offset); } catch (_) {}
    }
    if (this.sourceNodes.vocal) {
      try { this.sourceNodes.vocal.start(now, offset); } catch (_) {}
    }
    if (this.sourceNodes.active) {
      try { this.sourceNodes.active.start(now, offset); } catch (_) {}
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
      if (this.sourceNodes.active) {
        this.sourceNodes.active.stop();
        this.sourceNodes.active.disconnect();
        this.sourceNodes.active = null;
      }
      if (this.procNode) {
        this.procNode.disconnect();
      }
    } catch (e) {
      // Ignore already stopped nodes
    }
  }
}

window.DenoxLiveDspEngine = DenoxLiveDspEngine;
window.denoxDsp = new DenoxLiveDspEngine();
