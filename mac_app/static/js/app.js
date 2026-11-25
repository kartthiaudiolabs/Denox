// Copyright 2026 Inworld AI & Antigravity
// DENOX Application Controller

document.addEventListener('DOMContentLoaded', () => {
  const deck = new StemsAudioDeck();
  const specCanvas = document.getElementById('master-spectrum-canvas');
  if (specCanvas) {
    const visualizer = new MasterVisualizer('master-spectrum-canvas');
    visualizer.start();
  }

  let currentStemData = null;
  let activeTab = 'quantum';
  let liveRecording = false;
  let liveAudioCtx = null;
  let liveMediaStream = null;
  let liveProcessor = null;
  let liveSessionId = null;

  // --- TELEMETRY ---
  async function loadTelemetry() {
    try {
      const res = await fetch('/api/telemetry');
      const data = await res.json();
      
      const devicePill = document.getElementById('hud-device-pill');
      if (devicePill) {
        devicePill.innerHTML = `<span class="hud-dot"></span> ${data.device.toUpperCase()}`;
        if (data.is_mps) devicePill.classList.add('active');
      }

      const chipEl = document.getElementById('telem-chip');
      if (chipEl) chipEl.innerText = data.chip;

      const torchEl = document.getElementById('telem-torch');
      if (torchEl) torchEl.innerText = `${data.torch_version} (${data.precision})`;

      const rateEl = document.getElementById('telem-rate');
      if (rateEl) rateEl.innerText = `${data.sample_rate.toLocaleString()} Hz`;

      const osEl = document.getElementById('telem-os');
      if (osEl) osEl.innerText = data.os;
    } catch (e) {
      console.warn("Failed to load telemetry:", e);
    }
  }
  loadTelemetry();

  // --- TAB NAVIGATION ---
  document.querySelectorAll('.nav-tab').forEach(tabBtn => {
    tabBtn.addEventListener('click', () => {
      const target = tabBtn.getAttribute('data-tab');
      document.querySelectorAll('.nav-tab').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));

      tabBtn.classList.add('active');
      const pane = document.getElementById(`tab-${target}`);
      if (pane) pane.classList.add('active');
      activeTab = target;
    });
  });

  // --- FILE SELECTION & UPLOAD ---
  let selectedFilePath = null;
  const dropzone = document.getElementById('offline-dropzone');
  const fileInput = document.getElementById('offline-file-input');
  const selectedFileLabel = document.getElementById('selected-file-label');

  function setSelectedFile(path, filename, autoRun = false) {
    selectedFilePath = path;
    if (selectedFileLabel) {
      selectedFileLabel.innerHTML = `<strong>TARGET FILE:</strong> <span style="color:var(--logic-blue); font-weight:600;">${filename}</span>` + 
        (autoRun ? ` · <span style="color:var(--logic-cyan); font-weight:600;">APPLYING 5-MODULE DSP CHAIN...</span>` : ``);
    }
    const runBtn = document.getElementById('btn-run-isolation');
    if (runBtn) {
      runBtn.disabled = false;
      if (autoRun) {
        setTimeout(() => runBtn.click(), 50);
      }
    }
  }

  if (dropzone) {
    dropzone.addEventListener('click', () => fileInput && fileInput.click());
    dropzone.addEventListener('dragover', (e) => {
      e.preventDefault();
      dropzone.classList.add('dragover');
    });
    dropzone.addEventListener('dragleave', () => dropzone.classList.remove('dragover'));
    dropzone.addEventListener('drop', async (e) => {
      e.preventDefault();
      dropzone.classList.remove('dragover');
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadFile(e.dataTransfer.files[0]);
      }
    });
  }

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files && e.target.files[0]) {
        uploadFile(e.target.files[0]);
      }
    });
  }

  async function uploadFile(file) {
    const formData = new FormData();
    formData.append('file', file);
    try {
      if (selectedFileLabel) selectedFileLabel.innerHTML = `<strong>INSERTING:</strong> <span style="color:var(--logic-amber); font-weight:600;">${file.name}</span>...`;
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'success') {
        setSelectedFile(data.file_path, data.filename, true);
      }
    } catch (e) {
      alert("Failed to upload audio file: " + e.message);
    }
  }

  // --- DSP MASTERING SUITE & THRESHOLD STATE ---
  const dspState = {
    threshold: 1.0,
    denoiser: {
      enabled: true,
      preset: 'pure_isolated_vocal',
      threshold: 1.0,
      mode: 'spectral_gate',
      reduction_db: -60.0,
      ambient_bleed: 0.0
    },
    declicker: {
      enabled: true,
      preset: 'mouth_saliva_tame',
      sensitivity: 0.75,
      click_width_samples: 18,
      mode: 'mouth_clicks'
    },
    decrackler: {
      enabled: true,
      preset: 'vinyl_surface_gentle',
      sensitivity: 0.70,
      crackle_reduction: 0.65,
      frequency_band: 'high_frequency',
      density: 0.60
    },
    deverb: {
      enabled: true,
      preset: 'tight_vocal_booth',
      reduction_amount: 0.85,
      room_size: 'small',
      preserve_early_reflections: 0.40,
      spectral_floor_db: -22.0
    },
    deesser: {
      enabled: true,
      preset: 'broadcast_smooth_warmth',
      threshold_db: -22.0,
      reduction_amount: 0.65,
      frequency_focus_hz: 6500.0,
      bandwidth_q: 1.4,
      mode: 'split_band'
    }
  };

  let dspPresetsCache = null;
  async function loadDspPresets() {
    try {
      const res = await fetch('/api/dsp/presets');
      dspPresetsCache = await res.json();
      populateModuleDropdowns(dspPresetsCache);
    } catch (e) {
      console.warn("Failed to fetch DSP presets:", e);
    }
  }

  function populateModuleDropdowns(cache) {
    if (!cache) return;
    const modules = ['denoiser', 'declicker', 'decrackler', 'deverb', 'deesser'];
    modules.forEach(modKey => {
      const select = document.getElementById(`preset-${modKey}`);
      const presetsMap = cache[`${modKey}_presets`];
      if (!select || !presetsMap) return;

      const categories = {};
      Object.entries(presetsMap).forEach(([key, item]) => {
        const cat = item.category || 'General';
        if (!categories[cat]) categories[cat] = [];
        categories[cat].push({ key, ...item });
      });

      select.innerHTML = '';
      Object.entries(categories).forEach(([catName, items]) => {
        const group = document.createElement('optgroup');
        group.label = catName;
        items.forEach(it => {
          const opt = document.createElement('option');
          opt.value = it.key;
          opt.innerText = it.label || it.key;
          if (it.key === dspState[modKey].preset) opt.selected = true;
          group.appendChild(opt);
        });
        select.appendChild(group);
      });
    });
  }

  loadDspPresets();

  // Threshold Slider (De-Noiser Intensity)
  const thresholdSlider = document.getElementById('slider-isolation-threshold');
  const thresholdDisplay = document.getElementById('threshold-val-display');
  const dispDnthresh = document.getElementById('disp-dn-thresh');
  if (thresholdSlider) {
    thresholdSlider.addEventListener('input', (e) => {
      const val = parseInt(e.target.value, 10);
      dspState.threshold = val / 100.0;
      dspState.denoiser.threshold = val / 100.0;
      if (dispDnthresh) dispDnthresh.innerText = `${val}%`;
      if (typeof auditionManager !== 'undefined') {
        auditionManager.notifyParamChange('denoiser');
      }
      let desc = "ZERO BACKGROUND NOISE";
      if (val === 0) desc = "RAW AUDIO (0% CLEAN)";
      else if (val <= 25) desc = "SUBTLE NOISE ATTENUATION";
      else if (val <= 60) desc = "NATURAL SPEECH & ROOM AIR";
      else if (val < 100) desc = "WARM BALANCED SPEECH";
      if (thresholdDisplay) thresholdDisplay.innerText = `${val}% (${desc})`;
    });
  }

  // Module Toggles
  function setupModuleToggle(btnId, cardId, moduleKey) {
    const btn = document.getElementById(btnId);
    const card = document.getElementById(cardId);
    if (!btn || !card) return;
    btn.onclick = () => {
      const active = !dspState[moduleKey].enabled;
      dspState[moduleKey].enabled = active;
      btn.classList.toggle('active', active);
      btn.innerText = active ? "ACTIVE" : "BYPASSED";
      card.classList.toggle('module-active', active);
      card.classList.toggle('module-bypassed', !active);
    };
  }
  setupModuleToggle('btn-toggle-denoiser', 'card-module-denoiser', 'denoiser');
  setupModuleToggle('btn-toggle-declicker', 'card-module-declicker', 'declicker');
  setupModuleToggle('btn-toggle-decrackler', 'card-module-decrackler', 'decrackler');
  setupModuleToggle('btn-toggle-deverb', 'card-module-deverb', 'deverb');
  setupModuleToggle('btn-toggle-deesser', 'card-module-deesser', 'deesser');

  // Sliders Wiring
  function wireSlider(sliderId, dispId, callback, formatFn, modKey) {
    const slider = document.getElementById(sliderId);
    const disp = document.getElementById(dispId);
    if (!slider) return;
    slider.addEventListener('input', (e) => {
      const val = parseFloat(e.target.value);
      callback(val);
      if (disp) disp.innerText = formatFn ? formatFn(val) : `${val}%`;
      if (modKey && typeof auditionManager !== 'undefined') {
        auditionManager.notifyParamChange(modKey);
      }
    });
  }

  wireSlider('slider-dn-bleed', 'disp-dn-bleed', v => dspState.denoiser.ambient_bleed = v / 100.0, v => `${Math.round(v)}%`, 'denoiser');
  wireSlider('slider-dc-sens', 'disp-dc-sens', v => dspState.declicker.sensitivity = v / 100.0, null, 'declicker');
  wireSlider('slider-dc-width', 'disp-dc-width', v => dspState.declicker.click_width_samples = Math.round(v), v => `${Math.round(v)} smp`, 'declicker');
  wireSlider('slider-dcr-sens', 'disp-dcr-sens', v => dspState.decrackler.sensitivity = v / 100.0, null, 'decrackler');
  wireSlider('slider-dcr-reduc', 'disp-dcr-reduc', v => dspState.decrackler.crackle_reduction = v / 100.0, null, 'decrackler');
  wireSlider('slider-dv-reduc', 'disp-dv-reduc', v => dspState.deverb.reduction_amount = v / 100.0, null, 'deverb');
  wireSlider('slider-dv-early', 'disp-dv-early', v => dspState.deverb.preserve_early_reflections = v / 100.0, null, 'deverb');
  wireSlider('slider-de-reduc', 'disp-de-reduc', v => dspState.deesser.reduction_amount = v / 100.0, null, 'deesser');
  wireSlider('slider-de-freq', 'disp-de-freq', v => dspState.deesser.frequency_focus_hz = v, v => `${(v/1000).toFixed(1)} kHz`, 'deesser');

  // Presets Handlers
  const globalPresetSelect = document.getElementById('select-global-preset');
  if (globalPresetSelect) {
    globalPresetSelect.onchange = (e) => {
      const key = e.target.value;
      if (!dspPresetsCache || !dspPresetsCache.global_presets[key]) return;
      const p = dspPresetsCache.global_presets[key];

      // Update Sub-modules
      if (p.denoiser) applyModulePreset('denoiser', p.denoiser.preset, p.denoiser.enabled);
      if (p.declicker) applyModulePreset('declicker', p.declicker.preset, p.declicker.enabled);
      if (p.decrackler) applyModulePreset('decrackler', p.decrackler.preset, p.decrackler.enabled);
      if (p.deverb) applyModulePreset('deverb', p.deverb.preset, p.deverb.enabled);
      if (p.deesser) applyModulePreset('deesser', p.deesser.preset, p.deesser.enabled);
    };
  }

  function applyModulePreset(modKey, presetKey, enabled = true) {
    if (!dspPresetsCache) return;
    const presetsMap = dspPresetsCache[`${modKey}_presets`];
    if (!presetsMap || !presetsMap[presetKey]) return;
    const p = presetsMap[presetKey];

    const toggleBtn = document.getElementById(`btn-toggle-${modKey}`);
    const card = document.getElementById(`card-module-${modKey}`);
    const presetSelect = document.getElementById(`preset-${modKey}`);

    dspState[modKey].enabled = enabled;
    dspState[modKey].preset = presetKey;
    if (toggleBtn) {
      toggleBtn.classList.toggle('active', enabled);
      toggleBtn.innerText = enabled ? "ACTIVE" : "BYPASSED";
    }
    if (card) {
      card.classList.toggle('module-active', enabled);
      card.classList.toggle('module-bypassed', !enabled);
    }
    if (presetSelect) presetSelect.value = presetKey;

    if (modKey === 'denoiser') {
      dspState.denoiser.threshold = p.threshold;
      dspState.denoiser.mode = p.mode;
      dspState.denoiser.ambient_bleed = p.ambient_bleed || 0.0;
      dspState.threshold = p.threshold;
      const tSlider = document.getElementById('slider-isolation-threshold');
      if (tSlider) tSlider.value = Math.round(p.threshold * 100);
      const dThresh = document.getElementById('disp-dn-thresh');
      if (dThresh) dThresh.innerText = `${Math.round(p.threshold * 100)}%`;
      const bSlider = document.getElementById('slider-dn-bleed');
      if (bSlider) bSlider.value = Math.round((p.ambient_bleed || 0.0) * 100);
      const dBleed = document.getElementById('disp-dn-bleed');
      if (dBleed) dBleed.innerText = `${Math.round((p.ambient_bleed || 0.0) * 100)}%`;
      if (thresholdDisplay) {
        let desc = "ZERO BACKGROUND NOISE";
        const val = Math.round(p.threshold * 100);
        if (val === 0) desc = "RAW AUDIO (0% CLEAN)";
        else if (val <= 25) desc = "SUBTLE NOISE ATTENUATION";
        else if (val <= 60) desc = "NATURAL SPEECH & ROOM AIR";
        else if (val < 100) desc = "WARM BALANCED SPEECH";
        thresholdDisplay.innerText = `${val}% (${desc})`;
      }
    } else if (modKey === 'declicker') {
      dspState.declicker.sensitivity = p.sensitivity;
      dspState.declicker.click_width_samples = p.click_width_samples;
      dspState.declicker.mode = p.mode;
      const s1 = document.getElementById('slider-dc-sens');
      const s2 = document.getElementById('slider-dc-width');
      if (s1) s1.value = Math.round(p.sensitivity * 100);
      if (s2) s2.value = p.click_width_samples;
      const d1 = document.getElementById('disp-dc-sens');
      const d2 = document.getElementById('disp-dc-width');
      if (d1) d1.innerText = `${Math.round(p.sensitivity * 100)}%`;
      if (d2) d2.innerText = `${p.click_width_samples} smp`;
    } else if (modKey === 'decrackler') {
      dspState.decrackler.sensitivity = p.sensitivity;
      dspState.decrackler.crackle_reduction = p.crackle_reduction;
      dspState.decrackler.frequency_band = p.frequency_band;
      dspState.decrackler.density = p.density;
      const s1 = document.getElementById('slider-dcr-sens');
      const s2 = document.getElementById('slider-dcr-reduc');
      if (s1) s1.value = Math.round(p.sensitivity * 100);
      if (s2) s2.value = Math.round(p.crackle_reduction * 100);
      const d1 = document.getElementById('disp-dcr-sens');
      const d2 = document.getElementById('disp-dcr-reduc');
      if (d1) d1.innerText = `${Math.round(p.sensitivity * 100)}%`;
      if (d2) d2.innerText = `${Math.round(p.crackle_reduction * 100)}%`;
    } else if (modKey === 'deverb') {
      dspState.deverb.reduction_amount = p.reduction_amount;
      dspState.deverb.room_size = p.room_size;
      dspState.deverb.preserve_early_reflections = p.preserve_early_reflections;
      dspState.deverb.spectral_floor_db = p.spectral_floor_db;
      const s1 = document.getElementById('slider-dv-reduc');
      const s2 = document.getElementById('slider-dv-early');
      if (s1) s1.value = Math.round(p.reduction_amount * 100);
      if (s2) s2.value = Math.round(p.preserve_early_reflections * 100);
      const d1 = document.getElementById('disp-dv-reduc');
      const d2 = document.getElementById('disp-dv-early');
      if (d1) d1.innerText = `${Math.round(p.reduction_amount * 100)}%`;
      if (d2) d2.innerText = `${Math.round(p.preserve_early_reflections * 100)}%`;
    } else if (modKey === 'deesser') {
      dspState.deesser.reduction_amount = p.reduction_amount;
      dspState.deesser.frequency_focus_hz = p.frequency_focus_hz;
      dspState.deesser.threshold_db = p.threshold_db;
      dspState.deesser.bandwidth_q = p.bandwidth_q;
      dspState.deesser.mode = p.mode;
      const s1 = document.getElementById('slider-de-reduc');
      const s2 = document.getElementById('slider-de-freq');
      if (s1) s1.value = Math.round(p.reduction_amount * 100);
      if (s2) s2.value = p.frequency_focus_hz;
      const d1 = document.getElementById('disp-de-reduc');
      const d2 = document.getElementById('disp-de-freq');
      if (d1) d1.innerText = `${Math.round(p.reduction_amount * 100)}%`;
      if (d2) d2.innerText = `${(p.frequency_focus_hz/1000).toFixed(1)} kHz`;
    }

    if (typeof auditionManager !== 'undefined') {
      auditionManager.notifyParamChange(modKey);
    }
  }

  // Individual Module Preset dropdown changes
  ['denoiser', 'declicker', 'decrackler', 'deverb', 'deesser'].forEach(modKey => {
    const sel = document.getElementById(`preset-${modKey}`);
    if (sel) {
      sel.onchange = (e) => applyModulePreset(modKey, e.target.value, dspState[modKey].enabled);
    }
  });

  // --- INDEPENDENT MODULE AUDITION & LIVE PREVIEW CONTROLLER ---
  class AuditionManager {
    constructor() {
      this.modules = ['denoiser', 'declicker', 'decrackler', 'deverb', 'deesser'];
      this.activeMod = null;
      this.isPlaying = false;
      this.isBypassed = false;
      this.audioProc = new Audio();
      this.audioOrig = new Audio();
      this.debounceTimer = null;

      this.initEvents();
    }

    fmtTime(sec) {
      if (!sec || isNaN(sec)) return "00:00";
      const m = Math.floor(sec / 60);
      const s = Math.floor(sec % 60);
      return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`;
    }

    initEvents() {
      this.modules.forEach(mod => {
        const playBtn = document.getElementById(`btn-play-${mod}`);
        const abBtn = document.getElementById(`btn-ab-${mod}`);
        const seek = document.getElementById(`seek-${mod}`);

        if (playBtn) {
          playBtn.onclick = () => this.togglePreview(mod);
        }

        if (abBtn) {
          abBtn.onclick = () => this.toggleAB(mod);
        }

        if (seek) {
          seek.addEventListener('input', (e) => {
            const pct = parseFloat(e.target.value);
            this.seek(mod, pct);
          });
        }
      });

      this.audioProc.addEventListener('timeupdate', () => this.onTimeUpdate());
      this.audioProc.addEventListener('ended', () => this.onEnded());
      this.audioOrig.addEventListener('timeupdate', () => {
        if (this.isBypassed) this.onTimeUpdate();
      });
      this.audioOrig.addEventListener('ended', () => this.onEnded());
    }

    async togglePreview(mod) {
      if (this.activeMod === mod && this.isPlaying) {
        this.pause();
        return;
      }

      if (this.activeMod === mod && !this.isPlaying && this.audioProc.src) {
        this.play(mod);
        return;
      }

      this.stopAll();
      await this.loadAndPlay(mod);
    }

    async loadAndPlay(mod) {
      if (!selectedFilePath) {
        alert("Please insert or select an audio file first to audition.");
        return;
      }

      if (deck && deck.isPlaying) {
        deck.pause();
      }

      const playBtn = document.getElementById(`btn-play-${mod}`);
      const txt = document.getElementById(`txt-${mod}`);
      const icon = document.getElementById(`icon-${mod}`);
      const deckEl = document.getElementById(`audition-deck-${mod}`);

      if (playBtn) {
        playBtn.classList.add('loading');
        if (txt) txt.innerText = "LOADING...";
        if (icon) icon.innerText = "⏳";
      }

      try {
        const res = await fetch('/api/preview/module', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_path: selectedFilePath,
            module_name: mod,
            params: dspState[mod],
            stem_id: currentStemData?.stem_id,
            source_mode: 'original'
          })
        });

        if (!res.ok) throw new Error("Failed to load module preview");
        const data = await res.json();

        this.activeMod = mod;
        this.isBypassed = false;
        this.audioProc.src = data.preview_url;
        this.audioOrig.src = data.original_url;

        const abBtn = document.getElementById(`btn-ab-${mod}`);
        if (abBtn) {
          abBtn.classList.remove('bypassed');
          abBtn.innerText = "A (ACTIVE)";
        }

        await this.audioProc.play();
        this.isPlaying = true;

        if (playBtn) {
          playBtn.classList.remove('loading');
          playBtn.classList.add('playing');
          if (txt) txt.innerText = "PAUSE";
          if (icon) icon.innerText = "⏸";
        }
        if (deckEl) deckEl.classList.add('auditioning');

      } catch (err) {
        console.error("Preview error:", err);
        alert("Audition error: " + err.message);
        this.stopAll();
      }
    }

    play(mod) {
      this.isPlaying = true;
      const targetAudio = this.isBypassed ? this.audioOrig : this.audioProc;
      targetAudio.play();

      const playBtn = document.getElementById(`btn-play-${mod}`);
      const txt = document.getElementById(`txt-${mod}`);
      const icon = document.getElementById(`icon-${mod}`);
      const deckEl = document.getElementById(`audition-deck-${mod}`);

      if (playBtn) {
        playBtn.classList.add('playing');
        if (txt) txt.innerText = "PAUSE";
        if (icon) icon.innerText = "⏸";
      }
      if (deckEl) deckEl.classList.add('auditioning');
    }

    pause() {
      this.isPlaying = false;
      this.audioProc.pause();
      this.audioOrig.pause();

      if (this.activeMod) {
        const playBtn = document.getElementById(`btn-play-${this.activeMod}`);
        const txt = document.getElementById(`txt-${this.activeMod}`);
        const icon = document.getElementById(`icon-${this.activeMod}`);
        const deckEl = document.getElementById(`audition-deck-${this.activeMod}`);

        if (playBtn) {
          playBtn.classList.remove('playing');
          if (txt) txt.innerText = "PREVIEW";
          if (icon) icon.innerText = "▶";
        }
        if (deckEl) deckEl.classList.remove('auditioning');
      }
    }

    seek(mod, pct) {
      if (this.activeMod !== mod) return;
      const currentAudio = this.isBypassed ? this.audioOrig : this.audioProc;
      if (!currentAudio.duration) return;

      const targetTime = (pct / 100.0) * currentAudio.duration;
      this.audioProc.currentTime = targetTime;
      this.audioOrig.currentTime = targetTime;

      const timeEl = document.getElementById(`time-${mod}`);
      if (timeEl) {
        timeEl.innerText = `${this.fmtTime(targetTime)} / ${this.fmtTime(currentAudio.duration)}`;
      }
    }

    toggleAB(mod) {
      if (this.activeMod !== mod) return;
      this.isBypassed = !this.isBypassed;
      const abBtn = document.getElementById(`btn-ab-${mod}`);

      const curTime = this.isBypassed ? this.audioProc.currentTime : this.audioOrig.currentTime;
      if (this.isBypassed) {
        this.audioProc.pause();
        this.audioOrig.currentTime = curTime;
        if (this.isPlaying) this.audioOrig.play();
        if (abBtn) {
          abBtn.classList.add('bypassed');
          abBtn.innerText = "B (ORIGINAL)";
        }
      } else {
        this.audioOrig.pause();
        this.audioProc.currentTime = curTime;
        if (this.isPlaying) this.audioProc.play();
        if (abBtn) {
          abBtn.classList.remove('bypassed');
          abBtn.innerText = "A (ACTIVE)";
        }
      }
    }

    onTimeUpdate() {
      if (!this.activeMod) return;
      const currentAudio = this.isBypassed ? this.audioOrig : this.audioProc;
      const cur = currentAudio.currentTime || 0;
      const dur = currentAudio.duration || 1;

      const timeEl = document.getElementById(`time-${this.activeMod}`);
      const seek = document.getElementById(`seek-${this.activeMod}`);

      if (timeEl) {
        timeEl.innerText = `${this.fmtTime(cur)} / ${this.fmtTime(dur)}`;
      }
      if (seek) {
        seek.value = (cur / dur) * 100;
      }
    }

    onEnded() {
      this.pause();
      if (this.activeMod) {
        const seek = document.getElementById(`seek-${this.activeMod}`);
        if (seek) seek.value = 0;
      }
    }

    stopAll() {
      this.pause();
      this.modules.forEach(mod => {
        const playBtn = document.getElementById(`btn-play-${mod}`);
        const txt = document.getElementById(`txt-${mod}`);
        const icon = document.getElementById(`icon-${mod}`);
        const deckEl = document.getElementById(`audition-deck-${mod}`);
        const abBtn = document.getElementById(`btn-ab-${mod}`);

        if (playBtn) {
          playBtn.classList.remove('playing', 'loading');
          if (txt) txt.innerText = "PREVIEW";
          if (icon) icon.innerText = "▶";
        }
        if (deckEl) deckEl.classList.remove('auditioning');
        if (abBtn) {
          abBtn.classList.remove('bypassed');
          abBtn.innerText = "A (ACTIVE)";
        }
      });
      this.activeMod = null;
    }

    notifyParamChange(mod) {
      if (this.activeMod === mod && this.isPlaying) {
        clearTimeout(this.debounceTimer);
        this.debounceTimer = setTimeout(async () => {
          try {
            const curTime = this.isBypassed ? this.audioOrig.currentTime : this.audioProc.currentTime;
            const res = await fetch('/api/preview/module', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                file_path: selectedFilePath,
                module_name: mod,
                params: dspState[mod],
                stem_id: currentStemData?.stem_id,
                source_mode: 'original'
              })
            });
            if (res.ok) {
              const data = await res.json();
              this.audioProc.src = data.preview_url;
              this.audioProc.currentTime = curTime;
              if (!this.isBypassed && this.isPlaying) {
                this.audioProc.play();
              }
            }
          } catch (e) {
            console.warn("Live preview re-tune failed:", e);
          }
        }, 220);
      }
    }
  }

  const auditionManager = new AuditionManager();

  // --- ISOLATION EXECUTION ---
  const runBtn = document.getElementById('btn-run-isolation');
  const progressHud = document.getElementById('progress-hud');
  const progressBar = document.getElementById('cyber-progress-bar');
  const telemetryStats = document.getElementById('hud-stats-text');

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (!selectedFilePath) return;

      runBtn.disabled = true;
      runBtn.innerHTML = `5-MODULE INFERENCE & DSP MASTERING ACTIVE...`;
      if (progressHud) progressHud.style.display = 'block';
      if (progressBar) progressBar.style.width = '30%';

      const modelSelect = document.getElementById('select-model').value;
      const passesSelect = parseInt(document.getElementById('select-passes').value, 10);
      const precisionSelect = document.getElementById('select-precision').value;

      try {
        if (progressBar) progressBar.style.width = '65%';
        const startTime = Date.now();

        const res = await fetch('/api/isolate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            file_path: selectedFilePath,
            model: modelSelect,
            passes: passesSelect,
            precision: precisionSelect,
            threshold: dspState.threshold,
            dsp: dspState
          })
        });

        if (!res.ok) {
          const err = await res.json();
          throw new Error(err.detail || 'Isolation failed');
        }

        const data = await res.json();
        currentStemData = data;

        if (progressBar) progressBar.style.width = '100%';
        if (telemetryStats) {
          telemetryStats.innerText = `${data.duration}s audio in ${data.elapsed}s · ${data.rtf}x RTF (Threshold: ${Math.round(dspState.threshold*100)}%, Clicks Repaired: ${data.dsp_stats?.clicks_repaired || 0}, Crackle Attenuated: ${data.dsp_stats?.crackle_events || 0})`;
        }

        // Render Stems into Deck
        renderStems(data);
        deck.loadStems(data);

        // Auto play on success
        setTimeout(() => {
          deck.play();
        }, 300);

      } catch (err) {
        alert("Isolation Error: " + err.message);
      } finally {
        runBtn.disabled = false;
        runBtn.innerHTML = `INITIATE 5-MODULE DSP CHAIN`;
        setTimeout(() => {
          if (progressHud) progressHud.style.display = 'none';
          if (progressBar) progressBar.style.width = '0%';
        }, 2000);
      }
    });
  }

  // Instant DSP Re-Mastering Button
  const btnReapply = document.getElementById('btn-reapply-dsp');
  if (btnReapply) {
    btnReapply.onclick = async () => {
      if (!currentStemData || !currentStemData.stem_id) {
        alert("Please run isolation on an audio file first.");
        return;
      }

      btnReapply.disabled = true;
      btnReapply.innerText = "⚡ COMPUTING 5-MODULE CHAIN...";

      try {
        const res = await fetch('/api/remaster', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            stem_id: currentStemData.stem_id,
            dsp: dspState
          })
        });

        if (!res.ok) throw new Error("Remastering failed");
        const data = await res.json();

        // Update audio deck seamlessly
        const curTime = deck.tracks.vocal.audio.currentTime;
        const wasPlaying = deck.isPlaying;

        deck.tracks.vocal.audio.src = data.urls.vocal;
        deck.tracks.background.audio.src = data.urls.background;
        deck.tracks.vocal.audio.currentTime = curTime;
        deck.tracks.background.audio.currentTime = curTime;

        if (wasPlaying) deck.play();

        // Re-draw waveforms (Logic Pro colors)
        const cvVocal = document.getElementById('canvas-vocal');
        const cvBg = document.getElementById('canvas-bg');
        const progress = deck.duration > 0 ? curTime / deck.duration : 0;
        WaveformRenderer.drawWaveform(cvVocal, data.waveforms.vocal, '#2888e0', progress);
        WaveformRenderer.drawWaveform(cvBg, data.waveforms.background, '#f08c28', progress);

        currentStemData.waveforms.vocal = data.waveforms.vocal;
        currentStemData.waveforms.background = data.waveforms.background;

        btnReapply.innerText = `✓ RE-MASTERED (${data.dsp_stats.clicks_repaired} clicks, ${data.dsp_stats.crackle_events || 0} crackle events)`;
        setTimeout(() => {
          btnReapply.innerText = "⚡ RE-APPLY 5-MODULE DSP CHAIN (< 0.2s)";
          btnReapply.disabled = false;
        }, 1800);
      } catch (err) {
        alert("DSP Re-master Error: " + err.message);
        btnReapply.disabled = false;
        btnReapply.innerText = "⚡ RE-APPLY 5-MODULE DSP CHAIN (< 0.2s)";
      }
    };
  }


  function renderStems(data) {
    const stemsContainer = document.getElementById('stems-container');
    if (stemsContainer) stemsContainer.style.display = 'flex';

    // Render Canvas Waveforms (Logic Pro colors)
    const cvOrig = document.getElementById('canvas-orig');
    const cvVocal = document.getElementById('canvas-vocal');
    const cvBg = document.getElementById('canvas-bg');

    WaveformRenderer.drawWaveform(cvOrig, data.waveforms.original, '#8c92a0', 0);
    WaveformRenderer.drawWaveform(cvVocal, data.waveforms.vocal, '#2888e0', 0);
    WaveformRenderer.drawWaveform(cvBg, data.waveforms.background, '#f08c28', 0);

    // Setup interactive scrubbers on waveforms
    setupScrubber('canvas-orig-box', cvOrig, data.waveforms.original, '#8c92a0');
    setupScrubber('canvas-vocal-box', cvVocal, data.waveforms.vocal, '#2888e0');
    setupScrubber('canvas-bg-box', cvBg, data.waveforms.background, '#f08c28');

    // Download links
    document.getElementById('btn-dl-vocal').onclick = () => window.open(data.urls.vocal);
    document.getElementById('btn-dl-bg').onclick = () => window.open(data.urls.background);
    document.getElementById('btn-dl-orig').onclick = () => window.open(data.urls.original);

    // Reveal buttons
    document.getElementById('btn-reveal-stems').onclick = async () => {
      await fetch('/api/reveal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ path: data.paths.vocal })
      });
    };
  }

  function setupScrubber(boxId, canvas, peaks, color) {
    const box = document.getElementById(boxId);
    if (!box) return;
    box.onclick = (e) => {
      const rect = box.getBoundingClientRect();
      const progress = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
      deck.seek(progress * deck.duration);
    };
  }

  // --- TRANSPORT CONTROLS ---
  const btnMasterPlay = document.getElementById('btn-master-play');
  const timeDisplay = document.getElementById('time-display');

  if (btnMasterPlay) {
    btnMasterPlay.onclick = () => {
      if (typeof auditionManager !== 'undefined') auditionManager.pause();
      deck.togglePlay();
    };
  }

  deck.onStateChange = (isPlaying) => {
    if (btnMasterPlay) {
      btnMasterPlay.innerHTML = isPlaying ? '⏸' : '▶';
    }
  };

  deck.onTimeUpdate = (currentTime, duration) => {
    if (timeDisplay) {
      const cur = formatTime(currentTime);
      const dur = formatTime(duration);
      timeDisplay.innerText = `${cur} / ${dur}`;
    }

    const progress = duration > 0 ? currentTime / duration : 0;
    // Update playheads
    document.querySelectorAll('.playhead-scrubber').forEach(el => {
      el.style.left = `${progress * 100}%`;
    });

    if (currentStemData) {
      const cvOrig = document.getElementById('canvas-orig');
      const cvVocal = document.getElementById('canvas-vocal');
      const cvBg = document.getElementById('canvas-bg');
      WaveformRenderer.drawWaveform(cvOrig, currentStemData.waveforms.original, '#8c92a0', progress);
      WaveformRenderer.drawWaveform(cvVocal, currentStemData.waveforms.vocal, '#2888e0', progress);
      WaveformRenderer.drawWaveform(cvBg, currentStemData.waveforms.background, '#f08c28', progress);
    }
  };

  function formatTime(sec) {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    const ms = Math.floor((sec % 1) * 10);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}.${ms}`;
  }

  // --- A/B AUDITION MATRIX ---
  document.querySelectorAll('.ab-btn').forEach(btn => {
    btn.onclick = () => {
      document.querySelectorAll('.ab-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.getAttribute('data-mode');
      deck.setAuditionMode(mode);
      updateStemCardStates();
    };
  });

  function updateStemCardStates() {
    ['original', 'vocal', 'background'].forEach(k => {
      const card = document.getElementById(`card-${k}`);
      if (!card) return;
      card.classList.toggle('soloed', deck.tracks[k].solo);
      card.classList.toggle('muted', deck.tracks[k].mute);
    });
  }

  // Solo & Mute buttons
  ['original', 'vocal', 'background'].forEach(k => {
    const soloBtn = document.getElementById(`solo-${k}`);
    const muteBtn = document.getElementById(`mute-${k}`);
    if (soloBtn) {
      soloBtn.onclick = () => {
        const state = !deck.tracks[k].solo;
        deck.setSolo(k, state);
        soloBtn.classList.toggle('active-solo', state);
        updateStemCardStates();
      };
    }
    if (muteBtn) {
      muteBtn.onclick = () => {
        const state = !deck.tracks[k].mute;
        deck.setMute(k, state);
        muteBtn.classList.toggle('active-mute', state);
        updateStemCardStates();
      };
    }
  });

  // --- CYBER STREAM (LIVE MIC) ---
  const btnStartMic = document.getElementById('btn-start-mic');
  const btnStopMic = document.getElementById('btn-stop-mic');
  const streamStatus = document.getElementById('stream-status-pill');

  if (btnStartMic) {
    btnStartMic.onclick = async () => {
      try {
        liveSessionId = `live_${Date.now()}`;
        await fetch('/api/stream/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: liveSessionId, max_seconds: 60.0 })
        });

        liveMediaStream = await navigator.mediaDevices.getUserMedia({
          audio: { sampleRate: 48000, channelCount: 1, echoCancellation: false }
        });

        const AudioContext = window.AudioContext || window.webkitAudioContext;
        liveAudioCtx = new AudioContext({ sampleRate: 48000 });
        const source = liveAudioCtx.createMediaStreamSource(liveMediaStream);
        liveProcessor = liveAudioCtx.createScriptProcessor(4096, 1, 1);

        source.connect(liveProcessor);
        liveProcessor.connect(liveAudioCtx.destination);

        liveProcessor.onaudioprocess = async (e) => {
          if (!liveRecording) return;
          const inputData = e.inputBuffer.getChannelData(0);
          const pcm16 = new Int16Array(inputData.length);
          for (let i = 0; i < inputData.length; i++) {
            pcm16[i] = Math.max(-32768, Math.min(32767, Math.round(inputData[i] * 32767)));
          }

          // Base64 encode PCM
          const bytes = new Uint8Array(pcm16.buffer);
          let binary = '';
          for (let i = 0; i < bytes.byteLength; i++) binary += String.fromCharCode(bytes[i]);
          const base64 = btoa(binary);

          try {
            const res = await fetch('/api/stream/chunk', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                session_id: liveSessionId,
                pcm_base64: base64,
                rate: 48000
              })
            });
            const update = await res.json();
            if (update && update.waveform_output) {
              const liveVocalCv = document.getElementById('canvas-live-vocal');
              WaveformRenderer.drawWaveform(liveVocalCv, update.waveform_output, '#2888e0', 1.0);
            }
          } catch (err) {
            console.warn("Chunk error:", err);
          }
        };

        liveRecording = true;
        btnStartMic.disabled = true;
        btnStopMic.disabled = false;
        if (streamStatus) {
          streamStatus.innerHTML = `<span class="hud-dot"></span> STREAM ACTIVE (20ms LOOKAHEAD)`;
          streamStatus.classList.add('active');
        }
      } catch (err) {
        alert("Microphone Error: " + err.message);
      }
    };
  }

  if (btnStopMic) {
    btnStopMic.onclick = async () => {
      liveRecording = false;
      if (liveMediaStream) liveMediaStream.getTracks().forEach(t => t.stop());
      if (liveAudioCtx) liveAudioCtx.close();

      btnStartMic.disabled = false;
      btnStopMic.disabled = true;

      try {
        const res = await fetch('/api/stream/finish', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ session_id: liveSessionId })
        });
        const take = await res.json();
        alert(`Take captured: ${take.duration}s. Clean audio ready!`);
      } catch (err) {
        console.warn("Take finish error:", err);
      }

      if (streamStatus) {
        streamStatus.innerHTML = `<span class="hud-dot" style="background:#ffaa00;"></span> STREAM IDLE`;
        streamStatus.classList.remove('active');
      }
    };
  }

  // --- SYSTEM DIAGNOSTICS TAB ---
  const btnRunDoctor = document.getElementById('btn-run-doctor');
  const doctorOutput = document.getElementById('doctor-output');

  if (btnRunDoctor) {
    btnRunDoctor.onclick = async () => {
      btnRunDoctor.disabled = true;
      btnRunDoctor.innerText = "RUNNING VERIFICATION...";
      doctorOutput.innerText = "Querying system hardware and inspecting model checkpoints...\n";

      try {
        const res = await fetch('/api/doctor', { method: 'POST' });
        const data = await res.json();
        doctorOutput.innerText = data.output || JSON.stringify(data, null, 2);
      } catch (err) {
        doctorOutput.innerText = "Error running diagnostics: " + err.message;
      } finally {
        btnRunDoctor.disabled = false;
        btnRunDoctor.innerText = "RUN SYSTEM DIAGNOSTICS";
      }
    };
  }

  // --- NATIVE DIALOGS (PYWEBVIEW BRIDGES) ---
  const btnNativeBrowse = document.getElementById('btn-native-browse');
  if (btnNativeBrowse) {
    btnNativeBrowse.onclick = async () => {
      if (window.pywebview && window.pywebview.api && window.pywebview.api.select_file) {
        const path = await window.pywebview.api.select_file();
        if (path) {
          const name = path.split('/').pop();
          setSelectedFile(path, name, true);
        }
      } else {
        fileInput && fileInput.click();
      }
    };
  }

  // --- AUTOLOAD QUERY PARAMETER SUPPORT ---
  const urlParams = new URLSearchParams(window.location.search);
  const autoSample = urlParams.get('autoload');
  if (autoSample) {
    setTimeout(() => {
      setSelectedFile(`demo/examples/${autoSample}`, autoSample);
      if (urlParams.get('auto_run') === '1') {
        const rBtn = document.getElementById('btn-run-isolation');
        if (rBtn) rBtn.click();
      }
      const prevMod = urlParams.get('preview_mod');
      if (prevMod) {
        setTimeout(() => {
          const pBtn = document.getElementById(`btn-play-${prevMod}`);
          if (pBtn) pBtn.click();
        }, 400);
      }
    }, 600);
  }
});
