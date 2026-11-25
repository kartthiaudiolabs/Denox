---
name: run-demo
description: Launch the Inworld Voice Isolator demo with Offline file isolation and Online microphone recording. Use when someone wants to listen locally or explicitly asks for a temporary public demo.
---

# Run the demo

Follow [demo/README.md](../../../demo/README.md) for setup, commands and outputs.
It is the source of truth for demo operation; the demo extra pins Gradio 6.31.0.

1. Load both requested checkpoints, using the released offline and realtime
   models by default. `--offline-model` and `--online-model` select them
   independently. Registered models verify their checksums; confirm the requested
   file for a custom checkpoint. Both fixed modes must load successfully.
2. Use the bundled samples by default. `--examples ''` hides the samples.
3. Start locally. Add `--share` only when the user requested a public demo.
   Keep authentication in `VOICE_ISOLATION_DEMO_AUTH`. Microphone access needs
   localhost or HTTPS.
4. For a background server, write logs and the URL file under `staging/`. Report
   both checkpoints and the actual URL after startup succeeds.
5. Smoke-test Offline voice/background outputs and Online recording, live
   waveforms and spectrograms, stopping and playback, and the microphone input
   and isolated voice players a take leaves. Each online take is limited to 60
   seconds; capture pauses before playback. Check that the header names the
   expected device. Report when microphone or browser testing was unavailable.
