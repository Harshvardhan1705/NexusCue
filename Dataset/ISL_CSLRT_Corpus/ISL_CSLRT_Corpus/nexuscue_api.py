"""
NexusCue FastAPI bridge
Place this file beside create_sign_video.py.
Android will send recognized speech text to POST /api/generate.
"""

from pathlib import Path
from threading import Lock
from datetime import datetime, timezone
import os
import shutil
import subprocess
import sys
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
RENDERER_PATH = BASE_DIR / "create_sign_video.py"
OUTPUT_PATH = BASE_DIR / "sign_output.mp4"
GENERATED_DIR = BASE_DIR / "api_generated_videos"
GENERATED_DIR.mkdir(exist_ok=True)

RENDER_LOCK = Lock()

app = FastAPI(
    title="NexusCue API",
    description="Speech transcript to sign-video generation bridge",
    version="1.0.1",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Restrict before production deployment.
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
    expose_headers=["X-NexusCue-Mode", "X-NexusCue-Transcript"],
)


class GenerateRequest(BaseModel):
    sentence: str = Field(min_length=1, max_length=500)


@app.get("/")
def root():
    return {
        "app": "NexusCue API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health():
    return {
        "status": "ok" if RENDERER_PATH.is_file() else "renderer_missing",
        "renderer": RENDERER_PATH.name,
        "renderer_exists": RENDERER_PATH.is_file(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post(
    "/api/generate",
    response_class=FileResponse,
    responses={
        200: {
            "description": "Generated NexusCue sign video",
            "content": {"video/mp4": {}},
        }
    },
)
def generate_sign_video(request: GenerateRequest):
    sentence = request.sentence.strip()
    if not sentence:
        raise HTTPException(status_code=400, detail="Sentence cannot be blank.")

    if not RENDERER_PATH.is_file():
        raise HTTPException(
            status_code=500,
            detail=f"Renderer not found: {RENDERER_PATH.name}. Place nexuscue_api.py beside create_sign_video.py.",
        )

    with RENDER_LOCK:
        previous_mtime = (
            OUTPUT_PATH.stat().st_mtime_ns if OUTPUT_PATH.exists() else None
        )

        # Force UTF-8 in the child process. The renderer prints Unicode
        # characters such as arrows that Windows cp1252 cannot encode.
        child_env = os.environ.copy()
        child_env["PYTHONIOENCODING"] = "utf-8"
        child_env["PYTHONUTF8"] = "1"

        try:
            result = subprocess.run(
                [sys.executable, str(RENDERER_PATH)],
                input=sentence + "\n",
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                cwd=str(BASE_DIR),
                env=child_env,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            raise HTTPException(
                status_code=504,
                detail="Sign generation timed out after 5 minutes.",
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not start the sign renderer: {exc}",
            )

        if result.returncode != 0:
            diagnostic = (result.stdout or "") + "\n" + (result.stderr or "")
            raise HTTPException(
                status_code=500,
                detail={
                    "message": "Sign renderer failed.",
                    "return_code": result.returncode,
                    "output": diagnostic[-8000:],
                },
            )

        if not OUTPUT_PATH.is_file() or OUTPUT_PATH.stat().st_size == 0:
            raise HTTPException(
                status_code=500,
                detail="Renderer finished but sign_output.mp4 was not created.",
            )

        current_mtime = OUTPUT_PATH.stat().st_mtime_ns
        if previous_mtime is not None and current_mtime == previous_mtime:
            raise HTTPException(
                status_code=500,
                detail="Renderer did not update sign_output.mp4. Check the renderer output.",
            )

        video_name = f"nexcue_{uuid.uuid4().hex}.mp4"
        saved_video = GENERATED_DIR / video_name
        shutil.copy2(OUTPUT_PATH, saved_video)

        output_text = (result.stdout or "") + "\n" + (result.stderr or "")
        mode = (
            "english_fingerspelling_fallback"
            if "ENGLISH FINGERSPELLING FALLBACK" in output_text
            or "UNSUPPORTED SENTENCE" in output_text
            else "dataset_sentence_or_gloss"
        )

    return FileResponse(
        path=saved_video,
        media_type="video/mp4",
        filename=video_name,
        headers={
            "X-NexusCue-Mode": mode,
            "X-NexusCue-Transcript": sentence[:200],
        },
    )
