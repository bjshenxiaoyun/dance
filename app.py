#!/usr/bin/env python3
"""Minimal HTTP API wrapping the teacher/student dance comparison pipeline.

The pipeline itself is a batch job (extract poses, compare, render outputs),
but the deployment platform expects a long-running HTTP service with
liveness/readiness probes, so this exposes it over HTTP instead of only a CLI.
"""

from __future__ import annotations

import shutil
import sys
import threading
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from pipeline import run_comparison  # noqa: E402

OUTPUT_ROOT = ROOT / "output" / "api"

app = FastAPI(title="Dance Compare API")

# In-memory job status store. Good enough for a single-instance container;
# if the service ever scales to multiple replicas this needs a shared store
# (e.g. redis) since a status poll could land on a different instance than
# the one running the job.
JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()


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


class CompareByUrlRequest(BaseModel):
    teacher_url: str
    student_url: str
    max_frames: Optional[int] = None
    align: bool = True
    beats: bool = True
    meter: int = 4
    max_downbeats: int = 20


def _run_compare_job(job_id: str, req: CompareByUrlRequest) -> None:
    job_dir = OUTPUT_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    teacher_path = job_dir / "teacher.mp4"
    student_path = job_dir / "student.mp4"
    try:
        urllib.request.urlretrieve(req.teacher_url, teacher_path)
        urllib.request.urlretrieve(req.student_url, student_path)

        summary = run_comparison(
            teacher_path,
            student_path,
            job_dir / "result",
            max_frames=req.max_frames,
            align=req.align,
            use_beats=req.beats,
            meter=req.meter,
            max_downbeats=req.max_downbeats,
        )
        summary["job_id"] = job_id
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "done", "summary": summary}
    except Exception as e:
        with JOBS_LOCK:
            JOBS[job_id] = {"status": "error", "error": str(e)}


@app.post("/compare_by_url")
def compare_by_url(req: CompareByUrlRequest, background_tasks: BackgroundTasks) -> dict:
    """Same as /compare, but takes downloadable URLs instead of multipart file
    uploads — used by clients (e.g. the mini program) whose upload channel has
    a small request-body size limit, so the video itself has to go through
    object storage first and only the URL is sent here.

    Runs the actual comparison in the background and returns immediately,
    because the full pipeline (download + pose extraction + rendering) easily
    takes longer than the synchronous timeout of clients calling through the
    WeChat cloud-call gateway (wx.cloud.callContainer). Poll
    /jobs/{job_id}/status for completion."""
    job_id = uuid.uuid4().hex[:12]
    with JOBS_LOCK:
        JOBS[job_id] = {"status": "processing"}
    background_tasks.add_task(_run_compare_job, job_id, req)
    return {"job_id": job_id, "status": "processing"}


@app.get("/jobs/{job_id}/status")
def get_job_status(job_id: str) -> dict:
    with JOBS_LOCK:
        job = JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="job not found")
    return job


@app.get("/jobs/{job_id}/files/{rel_path:path}")
def get_job_file(job_id: str, rel_path: str) -> FileResponse:
    job_dir = (OUTPUT_ROOT / job_id / "result").resolve()
    file_path = (job_dir / rel_path).resolve()
    if not file_path.is_relative_to(job_dir) or not file_path.is_file():
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(file_path)
