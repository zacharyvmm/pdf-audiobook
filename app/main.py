"""
FastAPI web app: upload PDF → generate audiobook → download MP3s.
"""
from __future__ import annotations

import uuid
import shutil
import json
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .pipeline import extract_chapters, generate_audio, generate_audio_streaming, audio_to_mp3_bytes

app = FastAPI(title="PDF Audiobook")

# ── File-backed job store (survives restarts) ────────────────────

JOBS: dict[str, dict] = {}  # job_id → {status, chapters, ...}

BASE_DIR = Path(__file__).parent.parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


def _job_path(job_id: str) -> Path:
    return OUTPUT_DIR / job_id / "job.json"


def _save_job(job_id: str):
    (_job_path(job_id).parent).mkdir(parents=True, exist_ok=True)
    _job_path(job_id).write_text(json.dumps(JOBS[job_id], default=str))


def _load_job(job_id: str) -> dict | None:
    p = _job_path(job_id)
    if not p.is_file():
        return None
    return json.loads(p.read_text())


def _recover_jobs():
    """On startup, reload any existing job state from disk."""
    for job_dir in OUTPUT_DIR.iterdir():
        if not job_dir.is_dir():
            continue
        jf = job_dir / "job.json"
        if jf.is_file():
            try:
                job = json.loads(jf.read_text())
                JOBS[job_dir.name] = job
            except (json.JSONDecodeError, KeyError):
                pass


_recover_jobs()  # run at import time

# ── HTML frontend ────────────────────────────────────────────────

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>PDF Audiobook — Kokoro TTS</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: system-ui, -apple-system, sans-serif; max-width: 640px; margin: 40px auto; padding: 0 20px; color: #1a1a1a; background: #fafafa; }
  h1 { font-size: 1.5rem; margin-bottom: 8px; }
  .subtitle { color: #666; font-size: 0.9rem; margin-bottom: 24px; }

  /* ── Upload form ── */
  .upload-box { border: 2px dashed #d0d0d0; padding: 28px 24px; border-radius: 12px; text-align: center; background: #fff; transition: border-color 0.2s; }
  .upload-box.dragover { border-color: #1a73e8; background: #f0f6ff; }
  .upload-box.processing { border-style: solid; border-color: #e0e0e0; pointer-events: none; }
  input[type=file] { margin: 8px 0 16px; font-size: 0.95rem; }
  .options { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-bottom: 16px; }
  .options label { font-size: 0.85rem; color: #555; }
  select { margin-left: 4px; padding: 4px 8px; border-radius: 5px; border: 1px solid #bbb; font-size: 0.85rem; background: #fff; }

  /* ── Button ── */
  .btn { display: inline-flex; align-items: center; gap: 8px; padding: 10px 28px; font-size: 1rem; border-radius: 8px; border: none; cursor: pointer; font-weight: 500; transition: background 0.2s, opacity 0.2s; }
  .btn-primary { background: #1a73e8; color: #fff; }
  .btn-primary:hover { background: #1557b0; }
  .btn:disabled { opacity: 0.6; cursor: not-allowed; }

  /* ── Spinner ── */
  .spinner { display: none; width: 18px; height: 18px; border: 2.5px solid rgba(255,255,255,0.3); border-top-color: #fff; border-radius: 50%; animation: spin 0.7s linear infinite; }
  .btn.loading .spinner { display: inline-block; }
  .btn.loading .btn-text { opacity: 0.8; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* ── Progress ── */
  #progress-section { display: none; margin-top: 24px; }
  .progress-bar-outer { height: 8px; background: #e8e8e8; border-radius: 4px; overflow: hidden; }
  .progress-bar-inner { height: 100%; width: 0%; background: linear-gradient(90deg, #1a73e8, #4285f4); border-radius: 4px; transition: width 0.5s ease; }
  .status-text { margin-top: 10px; font-size: 0.9rem; color: #555; display: flex; align-items: center; gap: 6px; }
  .status-icon { font-size: 1.1rem; }
  .chapter-count { font-size: 0.8rem; color: #888; margin-left: auto; }

  /* ── Chapter cards ── */
  #chapters { margin-top: 20px; }
  .chapter-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; margin: 6px 0; background: #fff; border-radius: 8px; border: 1px solid #e8e8e8; animation: fadeIn 0.3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(-4px); } to { opacity: 1; transform: translateY(0); } }
  .chapter-card .ch-num { width: 28px; height: 28px; border-radius: 50%; background: #e8f0fe; color: #1a73e8; display: flex; align-items: center; justify-content: center; font-size: 0.8rem; font-weight: 600; flex-shrink: 0; }
  .chapter-card .ch-num.done { background: #e6f4ea; color: #1e8e3e; }
  .chapter-card .ch-info { flex: 1; min-width: 0; }
  .chapter-card .ch-title { font-size: 0.9rem; font-weight: 500; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .chapter-card .ch-meta { font-size: 0.75rem; color: #999; }
  .chapter-card .ch-dl { font-size: 0.8rem; color: #1a73e8; text-decoration: none; white-space: nowrap; font-weight: 500; flex-shrink: 0; }
  .chapter-card .ch-dl:hover { text-decoration: underline; }
  .chapter-card .ch-dl.pending { color: #bbb; pointer-events: none; }

  .done-banner { margin-top: 16px; padding: 12px 16px; background: #e6f4ea; border-radius: 8px; color: #1e8e3e; font-weight: 500; display: flex; align-items: center; gap: 8px; }
  .error-banner { margin-top: 16px; padding: 12px 16px; background: #fce8e6; border-radius: 8px; color: #c5221f; }
</style>
</head>
<body>
<h1>📖 PDF → Audiobook</h1>
<p class="subtitle">Upload a PDF book or article. Chapters are detected automatically — each becomes a separate audio file. Powered by <strong>Kokoro-82M</strong> TTS.</p>

<div class="upload-box" id="upload-box">
  <form id="upload-form" enctype="multipart/form-data">
    <input type="file" name="file" id="file-input" accept=".pdf" required>
    <div class="options">
      <label>🎤 <select name="voice">
        <option value="af_heart">af_heart (female, warm)</option>
        <option value="af_nova">af_nova (female, clear)</option>
        <option value="am_adam">am_adam (male)</option>
        <option value="bf_emma">bf_emma (British female)</option>
      </select></label>
      <label>⏱ <select name="speed">
        <option value="1.0">1.0×</option>
        <option value="1.2">1.2×</option>
        <option value="0.9">0.9×</option>
      </select></label>
    </div>
    <button type="submit" class="btn btn-primary" id="submit-btn">
      <span class="spinner"></span>
      <span class="btn-text">Generate Audiobook</span>
    </button>
  </form>
</div>

<div id="progress-section">
  <div class="progress-bar-outer"><div class="progress-bar-inner" id="progress-bar"></div></div>
  <div class="status-text">
    <span class="status-icon" id="status-icon">⏳</span>
    <span id="status-msg">Preparing...</span>
    <span class="chapter-count" id="chapter-count"></span>
  </div>
</div>

<div id="chapters"></div>

<script>
const form = document.getElementById('upload-form');
const submitBtn = document.getElementById('submit-btn');
const uploadBox = document.getElementById('upload-box');
const progressSection = document.getElementById('progress-section');
const progressBar = document.getElementById('progress-bar');
const statusIcon = document.getElementById('status-icon');
const statusMsg = document.getElementById('status-msg');
const chapterCount = document.getElementById('chapter-count');
const chaptersDiv = document.getElementById('chapters');

let pollTimer = null, knownChapterCount = 0, pollFailures = 0;

// ── Drag & drop styling ──
uploadBox.addEventListener('dragover', e => { e.preventDefault(); uploadBox.classList.add('dragover'); });
uploadBox.addEventListener('dragleave', () => uploadBox.classList.remove('dragover'));
uploadBox.addEventListener('drop', e => {
  e.preventDefault(); uploadBox.classList.remove('dragover');
  document.getElementById('file-input').files = e.dataTransfer.files;
});

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  const formData = new FormData(form);

  // ── Enter loading state ──
  submitBtn.classList.add('loading');
  submitBtn.disabled = true;
  uploadBox.classList.add('processing');
  progressSection.style.display = 'block';
  progressBar.style.width = '0%';
  statusIcon.textContent = '⏳';
  statusMsg.textContent = 'Uploading PDF...';
  chapterCount.textContent = '';
  chaptersDiv.innerHTML = '';
  knownChapterCount = 0;

  let res, data;
  try {
    res = await fetch('/api/generate', { method: 'POST', body: formData });
    data = await res.json();
  } catch (err) {
    showError('Cannot reach server. Is it running? (Error: ' + err.message + ')');
    return;
  }

  if (!res.ok) {
    showError(data.detail || 'Upload failed (HTTP ' + res.status + ')');
    return;
  }

  pollStatus(data.job_id);
});


function pollStatus(jobId) {
  clearInterval(pollTimer);
  pollFailures = 0;
  pollTimer = setInterval(async () => {
    try {
      const res = await fetch('/api/status/' + jobId);
      if (!res.ok) {
        pollFailures++;
        if (pollFailures > 10) {
          clearInterval(pollTimer);
          showError('Server returned ' + res.status + ' repeatedly. It may have restarted — please try uploading again.');
        }
        return;
      }
      pollFailures = 0;
      const data = await res.json();
      updateUI(data);

      if (data.status === 'done') {
        clearInterval(pollTimer);
        onDone(data);
      } else if (data.status === 'error') {
        clearInterval(pollTimer);
        showError(data.error);
      }
    } catch (err) {
      pollFailures++;
      if (pollFailures > 10) {
        clearInterval(pollTimer);
        showError('Lost connection to server. Please refresh and try again.');
      }
    }
  }, 800);
}

function updateUI(data) {
  progressBar.style.width = (data.progress || 0) + '%';
  statusMsg.textContent = data.status_text || 'Processing...';

  // Icon by status
  if (data.status === 'extracting') statusIcon.textContent = '📄';
  else if (data.status === 'generating' || data.chapters?.length > 0) statusIcon.textContent = '🔊';
  else statusIcon.textContent = '⏳';

  // Chapter counter with ETA
  let counterText = '';
  if (data.total_chapters) {
    counterText = data.chapters.length + ' / ' + data.total_chapters;
  }
  if (data.total_words && data.progress > 0 && data.progress < 100) {
    // ~25 words/sec processing on this CPU (2.4x realtime at ~150 wpm)
    const estTotalSec = data.total_words / 25;
    const estRemaining = Math.max(0, estTotalSec * (1 - data.progress / 100));
    if (estRemaining > 60) {
      counterText += (counterText ? ' · ~' : '~') + Math.round(estRemaining / 60) + ' min left';
    } else if (estRemaining > 5) {
      counterText += (counterText ? ' · ~' : '~') + Math.round(estRemaining) + 's left';
    }
  }
  chapterCount.textContent = counterText;

  // New chapters appeared — add cards
  if (data.chapters && data.chapters.length > knownChapterCount) {
    for (let i = knownChapterCount; i < data.chapters.length; i++) {
      const ch = data.chapters[i];
      const card = document.createElement('div');
      card.className = 'chapter-card';
      card.innerHTML = `
        <div class="ch-num">${i + 1}</div>
        <div class="ch-info">
          <div class="ch-title">${escapeHtml(ch.title)}</div>
          <div class="ch-meta">MP3 ready</div>
        </div>
        <a class="ch-dl" href="${ch.url}" download>⬇ Download</a>
      `;
      chaptersDiv.appendChild(card);
    }
    knownChapterCount = data.chapters.length;
  }

function onDone(data) {
  submitBtn.classList.remove('loading');
  submitBtn.disabled = false;
  submitBtn.querySelector('.btn-text').textContent = 'Generate Another';
  uploadBox.classList.remove('processing');
  progressBar.style.width = '100%';
  statusIcon.textContent = '✅';
  statusMsg.textContent = data.status_text || 'Done!';
  chapterCount.textContent = '';

  // Mark chapter numbers as done
  document.querySelectorAll('.ch-num').forEach(el => el.classList.add('done'));

  // Add done banner
  const banner = document.createElement('div');
  banner.className = 'done-banner';
  banner.innerHTML = '✅ All chapters generated! Click <strong>Download</strong> on each file above.';
  chaptersDiv.appendChild(banner);
}

function showError(msg) {
  clearInterval(pollTimer);
  submitBtn.classList.remove('loading');
  submitBtn.disabled = false;
  submitBtn.querySelector('.btn-text').textContent = 'Try Again';
  uploadBox.classList.remove('processing');
  progressSection.style.display = 'none';
  chaptersDiv.innerHTML = '<div class="error-banner">❌ ' + escapeHtml(msg || 'Unknown error') + '</div>';
}

function escapeHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML


@app.post("/api/generate")
async def generate(
    file: UploadFile = File(...),
    voice: str = Form("af_heart"),
    speed: float = Form(1.0),
):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        return JSONResponse({"detail": "Please upload a PDF file."}, status_code=400)

    job_id = uuid.uuid4().hex[:12]
    original_name = file.filename
    pdf_path = UPLOAD_DIR / f"{job_id}.pdf"
    job_dir = OUTPUT_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    with open(pdf_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    JOBS[job_id] = {
        "status": "extracting",
        "progress": 0,
        "status_text": "Extracting text from PDF...",
        "chapters": [],
        "total_chapters": 0,
        "error": None,
    }
    _save_job(job_id)

    import asyncio
    import functools

    # Run in thread pool to avoid blocking the event loop (Kokoro is CPU-bound)
    loop = asyncio.get_running_loop()
    loop.run_in_executor(
        None,
        functools.partial(_process_job_sync, job_id, str(pdf_path), job_dir, voice, speed, original_name),
    )

    return {"job_id": job_id}


@app.get("/api/status/{job_id}")
async def job_status(job_id: str):
    job = JOBS.get(job_id)
    if not job:
        # Try disk (survived a restart)
        job = _load_job(job_id)
        if job:
            JOBS[job_id] = job  # restore to memory
        else:
            return JSONResponse({"detail": "Job not found"}, status_code=404)
    return job


@app.get("/api/download/{job_id}/{filename}")
async def download_file(job_id: str, filename: str):
    file_path = OUTPUT_DIR / job_id / filename
    if not file_path.is_file():
        return JSONResponse({"detail": "File not found"}, status_code=404)
    return FileResponse(file_path, filename=filename, media_type="audio/mpeg")


def _process_job_sync(job_id: str, pdf_path: str, job_dir: Path, voice: str, speed: float, original_name: str = ""):
    try:
        JOBS[job_id]["status_text"] = "Extracting text from PDF..."
        chapters = extract_chapters(pdf_path, fallback_name=original_name)
        total = len(chapters)
        total_words = sum(len(t.split()) for _, t in chapters)
        JOBS[job_id]["total_chapters"] = total
        JOBS[job_id]["total_words"] = total_words
        JOBS[job_id]["progress"] = 5
        JOBS[job_id]["status"] = "generating"
        JOBS[job_id]["status_text"] = f"Found {total} chapter(s), ~{total_words} words. Generating audio..."

        for i, (title, text) in enumerate(chapters):
            JOBS[job_id]["status_text"] = f"Generating audio for: {title[:60]}..."
            JOBS[job_id]["progress"] = 5 + int(90 * (i / max(total, 1)))
            JOBS[job_id]["current_chapter"] = i + 1
            _save_job(job_id)

            # Calculate progress range for this chapter
            chapter_pct_start = 5 + int(90 * (i / max(total, 1)))
            chapter_pct_end = 5 + int(90 * ((i + 1) / max(total, 1)))

            def on_chunk(chunk_num: int, est_total: int):
                # Interpolate progress within this chapter
                frac = min(chunk_num / max(est_total, 1), 1.0)
                pct = int(chapter_pct_start + frac * (chapter_pct_end - chapter_pct_start))
                JOBS[job_id]["progress"] = pct
                JOBS[job_id]["status_text"] = (
                    f"Chapter {i+1}/{total}: {title[:50]}... "
                    f"(chunk {chunk_num}/{est_total})"
                )

            audio = generate_audio_streaming(
                text, voice=voice, speed=speed, progress_callback=on_chunk
            )
            if audio is None:
                continue

            mp3_bytes = audio_to_mp3_bytes(audio)

            safe = "".join(c for c in title if c.isalnum() or c in " _-")[:60].strip()
            safe = safe.replace(" ", "-").replace("_", "-")
            filename = f"ch_{i+1:02d}_{safe}.mp3"
            mp3_path = job_dir / filename
            mp3_path.write_bytes(mp3_bytes)

            JOBS[job_id]["chapters"].append({
                "title": title,
                "url": f"/api/download/{job_id}/{filename}",
            })
            _save_job(job_id)

        JOBS[job_id]["status"] = "done"
        JOBS[job_id]["progress"] = 100
        JOBS[job_id]["status_text"] = f"Done! {total} chapter(s) generated."
        _save_job(job_id)

    except Exception as e:
        JOBS[job_id]["status"] = "error"
        JOBS[job_id]["error"] = str(e)
        JOBS[job_id]["status_text"] = f"Error: {e}"
        _save_job(job_id)

    finally:
        # Clean up uploaded PDF
        Path(pdf_path).unlink(missing_ok=True)
