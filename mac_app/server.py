# Copyright 2026 Inworld AI & Antigravity
"""FastAPI server for the DENOX Mac application."""

import base64
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from mac_app import engine
from mac_app import dsp as dsp_module

logger = logging.getLogger("mac_app.server")

BASE_DIR = pathlib.Path(__file__).parent.resolve()
STATIC_DIR = BASE_DIR / "static"
REPO_ROOT = BASE_DIR.parent.resolve()
EXAMPLES_DIR = REPO_ROOT / "demo" / "examples"

app = FastAPI(title="DENOX // Pro 13 Audio Workstation", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IsolateRequest(BaseModel):
    file_path: str
    model: str = "offline"
    passes: int = 1
    precision: str = "auto"
    threshold: float = 1.0
    dsp: Optional[Dict[str, Any]] = None


class RemasterRequest(BaseModel):
    stem_id: str
    dsp: Dict[str, Any]


class ModulePreviewRequest(BaseModel):
    file_path: str
    module_name: str
    params: Dict[str, Any]
    stem_id: Optional[str] = None
    source_mode: str = "original"


class SampleRequest(BaseModel):
    sample_name: str


class ExportRequest(BaseModel):
    source_path: str
    target_path: str


class RevealRequest(BaseModel):
    path: str


class StreamChunkRequest(BaseModel):
    session_id: str
    pcm_base64: str
    rate: int = 48000


class StreamSessionRequest(BaseModel):
    session_id: str
    max_seconds: float = 60.0


@app.get("/api/telemetry")
async def get_telemetry():
    """Returns hardware, model, and system status."""
    try:
        return engine.get_system_telemetry()
    except Exception as e:
        logger.exception("Failed to get telemetry")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/samples")
async def list_samples():
    """Lists available bundled test samples."""
    samples = []
    if EXAMPLES_DIR.exists():
        for p in sorted(EXAMPLES_DIR.glob("sample_*.wav")):
            samples.append({
                "name": p.name,
                "path": str(p),
                "size_kb": round(p.stat().st_size / 1024, 1),
            })
    return {"samples": samples}


@app.post("/api/upload")
async def upload_audio(file: UploadFile = File(...)):
    """Receives drag & drop uploaded audio file and saves locally for processing."""
    try:
        temp_dir = engine.TMP_DIR / "uploads"
        temp_dir.mkdir(parents=True, exist_ok=True)
        dest_path = temp_dir / file.filename
        with dest_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        return {
            "status": "success",
            "file_path": str(dest_path),
            "filename": file.filename,
            "size": dest_path.stat().st_size
        }
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/raw")
@app.head("/api/audio/raw")
async def get_raw_audio(path: str):
    """Serves raw, example, or uploaded audio file for Web Audio decoding."""
    try:
        p = engine.resolve_source_audio_path(path)
        return FileResponse(path=str(p), media_type="audio/wav")
    except Exception:
        raise HTTPException(status_code=404, detail="Audio file not found")


@app.get("/api/dsp/presets")
async def get_dsp_presets():
    """Returns presets for Global Mastering, De-Noiser, De-Clicker, De-Crackler, De-Verb, and De-Esser."""
    return {
        "global_presets": dsp_module.GLOBAL_CHAIN_PRESETS,
        "denoiser_presets": dsp_module.DENOISER_PRESETS,
        "declicker_presets": dsp_module.DECLICKER_PRESETS,
        "decrackler_presets": dsp_module.DECRACKLER_PRESETS,
        "deverb_presets": dsp_module.DEVERB_PRESETS,
        "deesser_presets": dsp_module.DEESSER_PRESETS,
    }


@app.post("/api/isolate")
async def run_isolation(req: IsolateRequest):
    """Executes voice isolation on a file and applies the DSP chain."""
    try:
        dsp_dict = req.dsp or {}
        if "threshold" not in dsp_dict:
            dsp_dict["threshold"] = req.threshold

        result = engine.isolate_file(
            file_path=req.file_path,
            model_choice=req.model,
            offline_passes=req.passes,
            precision=req.precision,
            dsp_params=dsp_dict
        )
        return result
    except Exception as e:
        logger.exception("Isolation failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/remaster")
async def remaster_audio(req: RemasterRequest):
    """Re-applies DSP chain to cached isolation with sub-second turnaround."""
    try:
        result = engine.remaster_stem(req.stem_id, req.dsp)
        return result
    except Exception as e:
        logger.exception("Remaster failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/audio/{stem_id}/{filename}")
@app.head("/api/audio/{stem_id}/{filename}")
async def serve_audio(stem_id: str, filename: str):
    """Serves generated audio stems with Range support."""
    file_path = engine.TMP_DIR / stem_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Audio file not found")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


@app.post("/api/preview/module")
async def preview_module(req: ModulePreviewRequest):
    """Generates audition audio for an individual module with current settings."""
    try:
        result = engine.generate_module_preview(
            file_path=req.file_path,
            module_name=req.module_name,
            params=req.params,
            stem_id=req.stem_id,
            source_mode=req.source_mode,
        )
        return result
    except Exception as e:
        logger.exception("Module preview failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/preview/audio/{filename}")
@app.head("/api/preview/audio/{filename}")
async def serve_preview_audio(filename: str):
    """Serves generated module preview audio files with Range support."""
    file_path = engine.PREVIEWS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Preview audio not found")
    return FileResponse(path=str(file_path), media_type="audio/wav", filename=filename)


@app.post("/api/stream/start")
async def stream_start(req: StreamSessionRequest):
    """Initializes a live streaming microphone take."""
    success = engine.STREAM_MANAGER.create_session(req.session_id, req.max_seconds)
    if not success:
        raise HTTPException(status_code=500, detail="Could not create streaming session")
    return {"status": "ready", "session_id": req.session_id}


@app.post("/api/stream/chunk")
async def stream_chunk(req: StreamChunkRequest):
    """Pushes a chunk of audio to the live streaming session."""
    try:
        raw_bytes = base64.b64decode(req.pcm_base64)
        samples = np.frombuffer(raw_bytes, dtype=np.int16)
        update = engine.STREAM_MANAGER.push_chunk(req.session_id, samples, req.rate)
        if update is None:
            raise HTTPException(status_code=404, detail="Session not found or finished")
        return update
    except Exception as e:
        logger.exception("Streaming chunk processing failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/stream/finish")
async def stream_finish(req: StreamSessionRequest):
    """Finishes the streaming session and returns the recorded take stems."""
    result = engine.STREAM_MANAGER.finish_session(req.session_id)
    if not result:
        raise HTTPException(status_code=404, detail="Session not found")
    return result


@app.post("/api/reveal")
async def reveal_path(req: RevealRequest):
    """Reveals a file or directory in macOS Finder."""
    try:
        target = pathlib.Path(req.path).expanduser().resolve()
        if target.exists():
            subprocess.run(["open", "-R", str(target)], check=True)
            return {"status": "success", "revealed": str(target)}
        raise HTTPException(status_code=404, detail="Path does not exist")
    except Exception as e:
        logger.exception("Reveal in Finder failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/export")
async def export_audio(req: ExportRequest):
    """Exports a stem to a target location."""
    try:
        src = pathlib.Path(req.source_path).resolve()
        dst = pathlib.Path(req.target_path).resolve()
        if not src.exists():
            raise HTTPException(status_code=404, detail="Source file not found")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return {"status": "success", "exported_to": str(dst)}
    except Exception as e:
        logger.exception("Export failed")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/doctor")
async def run_doctor():
    """Runs the voice-isolate-doctor checks and returns detailed status."""
    try:
        # Run doctor in subprocess to capture exact text
        proc = subprocess.run(
            [sys.executable, "-m", "inworld.voice_isolation.doctor"],
            capture_output=True,
            text=True,
            check=False
        )
        output = proc.stdout.strip()
        if not output and proc.stderr:
            output = proc.stderr.strip()
        return {"status": "success", "output": output}
    except Exception as e:
        logger.exception("Doctor check failed")
        return {"status": "error", "error": str(e)}


# Mount static files
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    """Serves the main application UI."""
    return FileResponse(STATIC_DIR / "index.html")
