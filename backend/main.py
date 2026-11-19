import os, uuid, json, shutil, subprocess, re
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT = Path(os.getenv("CLIPGENIUS_DATA", "./data")).resolve()
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
for p in (UPLOADS, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ClipGenius AI API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
JOBS = {}

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError((result.stderr or result.stdout)[-3000:])
    return result.stdout

def duration_of(path: Path) -> float:
    raw = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(raw.strip())

def transcribe(path: Path):
    # Local open-source speech recognition. The first run downloads the model.
    from faster_whisper import WhisperModel
    model_name = os.getenv("WHISPER_MODEL", "small")
    device = os.getenv("WHISPER_DEVICE", "cpu")
    compute = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
    model = WhisperModel(model_name, device=device, compute_type=compute)
    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=False)
    return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments if s.text.strip()]

def rank_highlights(segments, total_duration, clip_seconds=35):
    # Heuristic ranking over transcript segments. Replace/augment with a
    # dedicated multimodal highlight model for production-quality ranking.
    cues = [
        "what", "no way", "oh my god", "omg", "bro", "wait", "crazy",
        "insane", "really", "why", "help", "laugh", "screaming", "wow",
        "unbelievable", "never", "look", "stop", "are you serious", "bruh",
        "shut up", "i can't", "lets go", "let's go", "no", "damn"
    ]
    windows = []
    if not segments:
        return []
    step = max(12, int(clip_seconds * 0.55))
    for start in range(0, max(1, int(total_duration)), step):
        end = min(total_duration, start + clip_seconds)
        selected = [s for s in segments if s["end"] > start and s["start"] < end]
        text = " ".join(s["text"] for s in selected).lower()
        if not text.strip():
            continue
        words = re.findall(r"\b[\w']+\b", text)
        cue_hits = sum(1 for cue in cues if cue in text)
        exclamations = text.count("!") + text.count("?")
        density = min(len(words) / max(1, (end-start)), 3.0)
        # Favor expressive dialogue and speech-rich windows.
        score = cue_hits * 2.2 + exclamations * 0.7 + density * 2.0
        windows.append({"start": float(start), "end": float(end), "score": round(score, 2),
                        "transcript": text[:900]})
    windows.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for item in windows:
        if all(abs(item["start"] - old["start"]) > clip_seconds * 0.65 for old in chosen):
            chosen.append(item)
        if len(chosen) >= 8:
            break
    return sorted(chosen, key=lambda x: x["start"])

def make_srt(segments, start, end, dest):
    def stamp(sec):
        ms = int(round(sec * 1000))
        h, ms = divmod(ms, 3600000)
        m, ms = divmod(ms, 60000)
        s, ms = divmod(ms, 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    rows, n = [], 1
    for seg in segments:
        if seg["end"] <= start or seg["start"] >= end:
            continue
        a, b = max(0, seg["start"]-start), min(end-start, seg["end"]-start)
        txt = seg["text"].replace("-->", "→").strip()
        if not txt:
            continue
        rows += [str(n), f"{stamp(a)} --> {stamp(max(a+0.15,b))}", txt, ""]
        n += 1
    dest.write_text("\n".join(rows), encoding="utf-8")

def render_clip(source, item, index, job_dir, segments, aspect="9:16"):
    start, end = item["start"], item["end"]
    srt = job_dir / f"clip_{index}.srt"
    make_srt(segments, start, end, srt)
    output = OUTPUTS / f"{job_dir.name}_clip_{index}.mp4"
    # Center crop to a vertical/square frame, scale to 1080 high, add readable captions.
    if aspect == "9:16":
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    elif aspect == "1:1":
        vf = "scale=1080:1080:force_original_aspect_ratio=increase,crop=1080:1080"
    else:
        vf = "scale=1920:1080:force_original_aspect_ratio=increase,crop=1920:1080"
    srt_filter = str(srt).replace("\\", "/").replace(":", r"\:")
    vf += f",subtitles='{srt_filter}':force_style='FontName=Arial,FontSize=20,Outline=2,Shadow=1,Alignment=2,MarginV=80'"
    run(["ffmpeg", "-y", "-ss", str(start), "-i", str(source), "-t", str(end-start),
         "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
         "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(output)])
    return {"id": index, "start": start, "end": end, "score": item["score"],
            "transcript": item["transcript"], "url": f"/media/{output.name}"}

def process_job(job_id, source, aspect):
    job = JOBS[job_id]
    try:
        job["status"] = "transcribing"
        job["progress"] = 15
        total = duration_of(source)
        segments = transcribe(source)
        job["progress"] = 45
        job["status"] = "finding_highlights"
        highlights = rank_highlights(segments, total)
        if not highlights:
            # Fallback: create a few evenly spaced segments if speech was not detected.
            length = min(30, total)
            highlights = [{"start": float(s), "end": float(min(total, s+length)),
                           "score": 0, "transcript": ""}
                          for s in [0, max(0, total/2-length/2), max(0,total-length)]]
        job["progress"] = 55
        job["status"] = "rendering"
        clips = []
        job_dir = ROOT / "jobs" / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        for i, item in enumerate(highlights, 1):
            clips.append(render_clip(source, item, i, job_dir, segments, aspect))
            job["progress"] = 55 + int(40*i/max(1,len(highlights)))
        job["clips"] = clips
        job["status"] = "completed"
        job["progress"] = 100
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
    finally:
        try: source.unlink(missing_ok=True)
        except Exception: pass

@app.get("/api/health")
def health():
    return {"ok": True, "name": "ClipGenius AI"}

@app.post("/api/jobs")
async def create_job(background_tasks: BackgroundTasks, video: UploadFile = File(...),
                     aspect: str = "9:16"):
    if aspect not in ("9:16", "1:1", "16:9"):
        raise HTTPException(400, "Unsupported aspect ratio")
    ext = Path(video.filename or "upload.mp4").suffix.lower()
    if ext not in (".mp4", ".mov", ".m4v", ".webm", ".mkv"):
        raise HTTPException(400, "Upload MP4, MOV, M4V, WebM or MKV")
    job_id = uuid.uuid4().hex[:12]
    source = UPLOADS / f"{job_id}{ext}"
    with source.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    JOBS[job_id] = {"id": job_id, "status": "queued", "progress": 0, "clips": []}
    background_tasks.add_task(process_job, job_id, source, aspect)
    return JOBS[job_id]

@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found (jobs are kept in memory in this starter)")
    return JOBS[job_id]

app.mount("/media", StaticFiles(directory=str(OUTPUTS)), name="media")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
