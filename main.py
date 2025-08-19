import io
import os
import time
import tempfile
from typing import Optional

import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
LANGUAGE = os.getenv("WHISPER_LANGUAGE", "es")
DEVICE = "cpu"  # On Render free tier typically CPU

app = FastAPI(title="Transcripto Whisper API", version="1.0.0")

# Lazy model load (first request)
_model = None

def get_model():
    global _model
    if _model is None:
        start = time.time()
        _model = whisper.load_model(MODEL_NAME, device=DEVICE)
        print(f"Loaded model {MODEL_NAME} in {time.time()-start:.2f}s")
    return _model

class TranscriptionResponse(BaseModel):
    duration: float
    text: str
    model: str
    language: Optional[str] = None

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing filename")
    filename_lower = file.filename.lower()
    if not any(filename_lower.endswith(ext) for ext in [".mp3", ".m4a", ".wav", ".ogg", ".flac", ".webm"]):
        raise HTTPException(status_code=400, detail="Unsupported file format")

    # Save to temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
            data = await file.read()
            tmp.write(data)
            tmp_path = tmp.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to write temp file: {e}")

    model = get_model()
    start = time.time()
    try:
        result = model.transcribe(tmp_path, language=LANGUAGE)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass

    duration = time.time() - start
    return TranscriptionResponse(duration=duration, text=result.get("text", "").strip(), model=MODEL_NAME, language=result.get("language"))

# Optional root
@app.get("/")
async def root():
    return {"name": "Transcripto Whisper API", "endpoints": ["/health", "/transcribe"], "model": MODEL_NAME}
