"""
PDF → Audiobook pipeline: extract text, detect chapters, TTS, MP3 output.
"""
from __future__ import annotations

import re
import subprocess
import tempfile
from pathlib import Path
from typing import Optional

import fitz  # pymupdf
import numpy as np
import soundfile as sf
from kokoro import KPipeline

# ── PDF extraction ──────────────────────────────────────────────

def extract_chapters(pdf_path: str | Path) -> list[tuple[str, str]]:
    """
    Extract text from a PDF, splitting by chapter headings.
    Returns list of (chapter_title, chapter_text).
    If no chapter headings found, returns one item = whole document.
    """
    doc = fitz.open(str(pdf_path))
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    chapter_pattern = re.compile(
        r'(?:^|\n)((?:Chapter|CHAPTER)\s+\d+[.:\s][^\n]*)',
        re.MULTILINE,
    )
    matches = list(chapter_pattern.finditer(full_text))

    if len(matches) >= 2:
        chapters = []
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start:end].strip()
            if body:
                chapters.append((title, body))
        return chapters

    # Single-item: article or chapterless book
    clean = full_text.strip()
    if not clean:
        return []
    title = doc.metadata.get("title") or Path(pdf_path).stem.replace("_", " ").title()
    return [(title, clean)]


# ── TTS ──────────────────────────────────────────────────────────

_pipeline: Optional[KPipeline] = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def generate_audio(text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray | None:
    """Run Kokoro TTS on text, return concatenated audio as numpy array (24kHz float32)."""
    pipeline = _get_pipeline()
    chunks = list(pipeline(text, voice=voice, speed=speed))
    if not chunks:
        return None
    return np.concatenate([audio for _, _, audio in chunks])


def audio_to_mp3_bytes(audio: np.ndarray, sample_rate: int = 24000) -> bytes:
    """Convert numpy audio array to MP3 bytes via ffmpeg pipe."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav_tmp:
        sf.write(wav_tmp.name, audio, sample_rate)
        wav_path = wav_tmp.name

    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", wav_path,
                "-codec:a", "libmp3lame", "-qscale:a", "2",
                "-f", "mp3", "pipe:1",
            ],
            capture_output=True,
            check=True,
        )
        return result.stdout
    finally:
        Path(wav_path).unlink(missing_ok=True)
