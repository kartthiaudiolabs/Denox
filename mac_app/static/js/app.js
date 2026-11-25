// Copyright 2026 Inworld AI & Antigravity
// Neo-Isolator Application Controller

document.addEventListener('DOMContentLoaded', () => {
  const deck = new StemsAudioDeck();
  const visualizer = new MasterVisualizer('master-spectrum-canvas');
  visualizer.start();

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

  // --- PRESETS / SAMPLES ---
  async function loadPresetSamples() {
    try {
      const res = await fetch('/api/samples');
      const data = await res.json();
      const container = document.getElementById('presets-list');
      if (!container) return;
      container.innerHTML = '';

      data.samples.slice(0, 8).forEach(sample => {
        const chip = document.createElement('div');
        chip.className = 'preset-chip';
        chip.innerText = sample.name;
        chip.addEventListener('click', () => {
          setSelectedFile(sample.path, sample.name);
        });
        container.appendChild(chip);
      });
    } catch (e) {
      console.warn("Failed to load sample presets:", e);
    }
  }
  loadPresetSamples();

  // --- FILE SELECTION & UPLOAD ---
  let selectedFilePath = null;
  const dropzone = document.getElementById('offline-dropzone');
  const fileInput = document.getElementById('offline-file-input');
  const selectedFileLabel = document.getElementById('selected-file-label');

  function setSelectedFile(path, filename) {
    selectedFilePath = path;
    if (selectedFileLabel) {
      selectedFileLabel.innerHTML = `<strong>TARGET FILE:</strong> <span style="color:var(--cyan-neon);">${filename}</span>`;
    }
    const runBtn = document.getElementById('btn-run-isolation');
    if (runBtn) runBtn.disabled = false;
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
      if (selectedFileLabel) selectedFileLabel.innerText = `Uploading ${file.name}...`;
      const res = await fetch('/api/upload', { method: 'POST', body: formData });
      const data = await res.json();
      if (data.status === 'success') {
        setSelectedFile(data.file_path, data.filename);
      }
    } catch (e) {
      alert("Failed to upload audio file: " + e.message);
    }
  }

  // --- ISOLATION EXECUTION ---
  const runBtn = document.getElementById('btn-run-isolation');
  const progressHud = document.getElementById('progress-hud');
  const progressBar = document.getElementById('cyber-progress-bar');
  const telemetryStats = document.getElementById('hud-stats-text');

  if (runBtn) {
    runBtn.addEventListener('click', async () => {
      if (!selectedFilePath) return;

      runBtn.disabled = true;
      runBtn.innerHTML = `NEURAL INFERENCE ACTIVE...`;
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
            precision: precisionSelect
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
          telemetryStats.innerText = `${data.duration}s audio in ${data.elapsed}s · ${data.rtf}x Real-Time Factor (MPS Accelerated)`;
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
        runBtn.innerHTML = `INITIATE NEURAL ISOLATION`;
        setTimeout(() => {
          if (progressHud) progressHud.style.display = 'none';
          if (progressBar) progressBar.style.width = '0%';
        }, 2000);
      }
    });
  }

  function renderStems(data) {
    const stemsContainer = document.getElementById('stems-container');
    if (stemsContainer) stemsContainer.style.display = 'flex';

    // Render Canvas Waveforms
    const cvOrig = document.getElementById('canvas-orig');
    const cvVocal = document.getElementById('canvas-vocal');
    const cvBg = document.getElementById('canvas-bg');

    WaveformRenderer.drawWaveform(cvOrig, data.waveforms.original, '#7f90a6', 0);
    WaveformRenderer.drawWaveform(cvVocal, data.waveforms.vocal, '#00f0ff', 0);
    WaveformRenderer.drawWaveform(cvBg, data.waveforms.background, '#9d00ff', 0);

    // Setup interactive scrubbers on waveforms
    setupScrubber('canvas-orig-box', cvOrig, data.waveforms.original, '#7f90a6');
    setupScrubber('canvas-vocal-box', cvVocal, data.waveforms.vocal, '#00f0ff');
    setupScrubber('canvas-bg-box', cvBg, data.waveforms.background, '#9d00ff');

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
    btnMasterPlay.onclick = () => deck.togglePlay();
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
      WaveformRenderer.drawWaveform(cvOrig, currentStemData.waveforms.original, '#7f90a6', progress);
      WaveformRenderer.drawWaveform(cvVocal, currentStemData.waveforms.vocal, '#00f0ff', progress);
      WaveformRenderer.drawWaveform(cvBg, currentStemData.waveforms.background, '#9d00ff', progress);
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
              WaveformRenderer.drawWaveform(liveVocalCv, update.waveform_output, '#00f0ff', 1.0);
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
          setSelectedFile(path, name);
        }
      } else {
        fileInput && fileInput.click();
      }
    };
  }
});
