# Voice Isolator demo

Two tabs share one page, with both models loaded before the demo opens:

- **Offline:** upload a file, choose a sample or record a clip, then isolate its
  voice and background. Each output has a player, spectrogram and download.
- **Online:** start the microphone and watch input and isolated-voice waveforms
  and spectrograms update as you speak. Each take lasts up to 60 seconds.
  **Pause & listen** stops capture and plays the completed clean recording;
  **Stop** finishes it without playback. Below the panel, the microphone input
  as recorded and the voice isolated from it play side by side, sample-aligned,
  each with a download. Start a new recording for another take.
  Playing audio on either tab pauses capture first to avoid recording playback.

The header names the device inference runs on: CPU, Metal GPU (Apple
silicon) or CUDA GPU. The speed readouts on both tabs repeat it.

The offline background is the aligned 48 kHz mono input minus the isolated
voice. It can contain voice leakage. Downloads save the players' 16-bit WAVs
without normalizing their level; peaks above full scale are clipped. The CLI
writes float32 voice/background WAVs when those peaks need to be preserved.

## Run locally

From a repository checkout:

```bash
pip install -e ".[demo]"  # includes Gradio 6.31.0
python demo/app.py
```

Open `http://127.0.0.1:7860`. Microphone access requires localhost or HTTPS.
The default models download from `inworld/voice-isolator` on first use.
Offline uses `voice-isolator-v1`; Online uses `voice-isolator-v1-realtime`
with 20 ms of look-ahead. Each tab keeps its own model loaded in its fixed mode
and names it: they are different models, so the same audio gives slightly
different results on the two tabs.

To use local checkpoints:

```bash
python demo/app.py --offline-model offline.safetensors --online-model realtime.safetensors
```

These options also accept registry names and `hf://` locations. Each checkpoint
must support its tab's mode. `--device` and `--precision` apply to both models.
Live operation needs a device that processes audio faster than it arrives.
If the device or connection falls behind, capture pauses at roughly four
seconds of pending audio and finishes the recording. Startup runs half a second
of silence through both models, so the first take does not wait for the GPU to
compile its kernels.

Limits, for a demo shared with a few people:

- The Offline tab takes up to 10 minutes of audio per input and uploads up to
  300 MB.
- At most two live takes run at once; a third is asked to try again shortly.
  Closing or reloading a tab releases its take at once.
- On an Apple GPU both models share the device, which runs one task at a time.
  While an Offline isolation runs, a new live take is refused and a take in
  progress waits; a take kept waiting more than about 45 seconds stops.
See the [model card](../MODEL_CARD.md) for model details.

- `--examples ''` hides samples; uploads and recording remain available.
- `--port 7861` changes the port. The default host is loopback; use
  `--server-name 0.0.0.0` for access from other machines.
- `--share` creates a temporary public Gradio URL. `--url-file staging/demo_url.txt`
  records the local and share URLs; create `staging/` first.
- `VOICE_ISOLATION_DEMO_AUTH=user:password` enables authentication.
- `VOICE_ISOLATION_OFFLINE_MODEL` and `VOICE_ISOLATION_ONLINE_MODEL` set the two
  default checkpoints; `VOICE_ISOLATION_EXAMPLES` sets the samples directory.
