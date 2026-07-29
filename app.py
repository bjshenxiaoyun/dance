#!/usr/bin/env python3
"""Minimal HTTP API wrapping the teacher/student dance comparison pipeline.

The pipeline itself is a batch job (extract poses, compare, render outputs),
but the deployment platform expects a long-running HTTP service with
liveness/readiness probes, so this exposes it over HTTP instead of only a CLI.
"""

from __future__ import annotations

import shutil
import sys
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_comparison  # noqa: E402

OUTPUT_ROOT = ROOT / "output" / "api"

app = FastAPI(title="Dance Compare API")


@app.get("/")
@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/compare")
def compare(
    teacher: UploadFile = File(...),
    student: UploadFile = File(...),
    max_frames: Optional[int] = Form(None),
    align: bool = Form(True),
    beats: bool = Form(True),
    meter: int = Form(4),
    max_downbeats: int = Form(20),
) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = job_dir / f"teacher{Path(teacher.filename or 'teacher.mp4').suffix}"
    student_path = job_dir / f"student{Path(student.filename or 'student.mp4').suffix}"
    with teacher_path.open("wb") as f:
        shutil.copyfileobj(teacher.file, f)
    with student_path.open("wb") as f:
        shutil.copyfileobj(student.file, f)

    try:
        summary = run_comparison(
            teacher_path,
            student_path,
            job_dir / "result",
            max_frames=max_frames,
            align=align,
            use_beats=beats,
            meter=meter,
            max_downbeats=max_downbeats,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    summary["job_id"] = job_id
    return summary


@app.get("/jobs/{job_id}/files/{rel_path:path}")
def get_job_file(job_id: str, rel_path: str) -> FileResponse:
    job_dir = (OUTPUT_ROOT / job_id / "result").resolve()
    file_path = (job_dir / rel_path).resolve()
    if not file_path.is_relative_to(job_dir) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(file_path)
