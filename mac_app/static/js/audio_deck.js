// Copyright 2026 Inworld AI & Antigravity
// Multi-Stem Synchronized Audio Deck

class StemsAudioDeck {
  constructor() {
    this.audioCtx = null;
    this.masterGain = null;
    this.tracks = {
      original: { audio: new Audio(), gainNode: null, peaks: [], solo: false, mute: false, volume: 1.0 },
      vocal: { audio: new Audio(), gainNode: null, peaks: [], solo: false, mute: false, volume: 1.0 },
      background: { audio: new Audio(), gainNode: null, peaks: [], solo: false, mute: false, volume: 1.0 },
    };
    this.duration = 0;
    this.isPlaying = false;
    this.isLooping = false;
    this.auditionMode = 'vocal'; // 'vocal', 'background', 'original', 'custom'
    this.onTimeUpdate = null;
    this.onStateChange = null;

    this._initAudioElements();
  }

  _ensureContext() {
    if (!this.audioCtx) {
      const AudioContext = window.AudioContext || window.webkitAudioContext;
      this.audioCtx = new AudioContext();
      this.masterGain = this.audioCtx.createGain();
      this.masterGain.connect(this.audioCtx.destination);

      for (const [key, track] of Object.entries(this.tracks)) {
        const source = this.audioCtx.createMediaElementSource(track.audio);
        track.gainNode = this.audioCtx.createGain();
        source.connect(track.gainNode);
        track.gainNode.connect(this.masterGain);
      }
    }
    if (this.audioCtx.state === 'suspended') {
      this.audioCtx.resume();
    }
  }

  _initAudioElements() {
    for (const [key, track] of Object.entries(this.tracks)) {
      track.audio.crossOrigin = 'anonymous';
      track.audio.preload = 'auto';

      if (key === 'vocal') {
        track.audio.addEventListener('timeupdate', () => {
          if (this.onTimeUpdate) {
            this.onTimeUpdate(track.audio.currentTime, this.duration);
          }
        });
        track.audio.addEventListener('ended', () => {
          if (this.isLooping) {
            this.seek(0);
            this.play();
          } else {
            this.pause();
            this.seek(0);
          }
        });
      }
    }
  }

  loadStems(stemData) {
    this.pause();
    this.duration = stemData.duration || 0;

    for (const key of ['original', 'vocal', 'background']) {
      const url = stemData.urls[key];
      const peaks = stemData.waveforms ? stemData.waveforms[key] : [];
      this.tracks[key].audio.src = url;
      this.tracks[key].audio.load();
      this.tracks[key].peaks = peaks;
      this.tracks[key].solo = false;
      this.tracks[key].mute = false;
    }

    // Default to vocal audition
    this.setAuditionMode('vocal');
  }

  play() {
    this._ensureContext();
    const vocalAudio = this.tracks.vocal.audio;
    const targetTime = vocalAudio.currentTime;

    // Resync all
    for (const track of Object.values(this.tracks)) {
      track.audio.currentTime = targetTime;
      track.audio.play().catch(e => console.warn("Autoplay blocked or waiting:", e));
    }
    this.isPlaying = true;
    if (this.onStateChange) this.onStateChange(true);
  }

  pause() {
    for (const track of Object.values(this.tracks)) {
      track.audio.pause();
    }
    this.isPlaying = false;
    if (this.onStateChange) this.onStateChange(false);
  }

  togglePlay() {
    if (this.isPlaying) this.pause();
    else this.play();
  }

  seek(seconds) {
    for (const track of Object.values(this.tracks)) {
      track.audio.currentTime = seconds;
    }
    if (this.onTimeUpdate) {
      this.onTimeUpdate(seconds, this.duration);
    }
  }

  setMasterVolume(val) {
    if (this.masterGain) {
      this.masterGain.gain.setValueAtTime(val, this.audioCtx.currentTime);
    }
  }

  setSolo(trackKey, state) {
    this.tracks[trackKey].solo = state;
    this.auditionMode = 'custom';
    this._updateGains();
  }

  setMute(trackKey, state) {
    this.tracks[trackKey].mute = state;
    this.auditionMode = 'custom';
    this._updateGains();
  }

  setRealtimeThreshold(threshold) {
    if (this.auditionMode !== 'vocal' || !this.audioCtx) return;
    const t = Math.max(0.0, Math.min(1.0, threshold));
    const now = this.audioCtx.currentTime;
    if (this.tracks.vocal.gainNode) {
      this.tracks.vocal.gainNode.gain.setTargetAtTime(t * this.tracks.vocal.volume, now, 0.005);
    }
    if (this.tracks.original.gainNode) {
      this.tracks.original.gainNode.gain.setTargetAtTime((1.0 - t) * this.tracks.original.volume, now, 0.005);
    }
  }

  setAuditionMode(mode) {
    this.auditionMode = mode;
    if (mode === 'vocal') {
      this.tracks.vocal.solo = true;
      this.tracks.vocal.mute = false;
      this.tracks.original.solo = false;
      this.tracks.original.mute = false; // Allow real-time threshold crossfade with vocal
      this.tracks.background.solo = false;
      this.tracks.background.mute = true;
    } else if (mode === 'background') {
      this.tracks.background.solo = true;
      this.tracks.background.mute = false;
      this.tracks.vocal.solo = false;
      this.tracks.vocal.mute = true;
      this.tracks.original.solo = false;
      this.tracks.original.mute = true;
    } else if (mode === 'original') {
      this.tracks.original.solo = true;
      this.tracks.original.mute = false;
      this.tracks.vocal.solo = false;
      this.tracks.vocal.mute = true;
      this.tracks.background.solo = false;
      this.tracks.background.mute = true;
    } else if (mode === 'all') {
      for (const t of Object.values(this.tracks)) {
        t.solo = false;
        t.mute = false;
      }
    }
    this._updateGains();
  }

  _updateGains() {
    this._ensureContext();
    const hasAnySolo = Object.values(this.tracks).some(t => t.solo);

    for (const [key, track] of Object.entries(this.tracks)) {
      if (!track.gainNode) continue;
      let targetGain = track.volume;

      if (track.mute) {
        targetGain = 0;
      } else if (hasAnySolo) {
        if (this.auditionMode === 'vocal' && (key === 'vocal' || key === 'original')) {
          const t = Math.max(0.0, Math.min(1.0, window.dspState?.threshold ?? 1.0));
          targetGain = (key === 'vocal' ? t : (1.0 - t)) * track.volume;
        } else {
          targetGain = track.solo ? track.volume : 0;
        }
      }

      track.gainNode.gain.setTargetAtTime(targetGain, this.audioCtx.currentTime, 0.005);
    }
  }
}

window.StemsAudioDeck = StemsAudioDeck;
