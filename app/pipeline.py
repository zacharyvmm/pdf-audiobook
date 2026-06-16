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

def extract_chapters(pdf_path: str | Path, fallback_name: str = "") -> list[tuple[str, str]]:
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
    title = doc.metadata.get("title") or fallback_name or Path(pdf_path).stem.replace("_", " ").title()
    return [(title, clean)]


# ── TTS ──────────────────────────────────────────────────────────

_pipeline: Optional[KPipeline] = None


def _get_pipeline() -> KPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    return _pipeline


def normalize_text(text: str) -> str:
    """
    Join broken PDF lines into flowing paragraphs.
    Handles: hyphenated line breaks, mid-sentence line breaks,
    and preserves paragraph breaks (double newlines).
    """
    lines = text.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            if result and result[-1] != "\n":
                result.append("\n")  # paragraph break
            continue
        if result and result[-1] != "\n":
            prev = result[-1].rstrip()
            # Hyphenated word break: join "convolu-\ntional" → "convolutional"
            # Only join the hyphenated word, not the entire lines.
            if prev.endswith("-"):
                # Find the last word in prev (the hyphenated fragment)
                m = re.search(r"(\w+)-$", prev)
                if m and stripped and stripped[0].islower():
                    frag = m.group(1)  # "convolu" or "sequence"
                    # Find the continuation in stripped (first word)
                    cont_match = re.match(r"(\w+)", stripped)
                    if cont_match:
                        cont = cont_match.group(1)
                        # If the fragment is a common prefix or a recognizable word,
                        # the hyphen is intentional (compound). Keep it.
                        COMMON_PREFIXES = frozenset({
                            "pre", "re", "un", "non", "over", "under", "inter",
                            "intra", "extra", "semi", "anti", "multi", "post",
                            "cross", "micro", "macro", "pseudo", "neo", "proto",
                            "hyper", "hypo", "ultra", "infra", "counter", "mid",
                            "mini", "maxi", "super", "mega", "giga", "nano",
                        })
                        # Common English words — if the fragment is a real word,
                        # the hyphen is intentional (compound like "sequence-aligned")
                        COMMON_WORDS = frozenset({
                            "about", "above", "across", "after", "again", "against",
                            "almost", "alone", "along", "already", "also", "although",
                            "always", "among", "another", "any", "anyone", "anything",
                            "around", "available", "away", "back", "because", "become",
                            "been", "before", "behind", "being", "below", "between",
                            "both", "bring", "came", "cannot", "change", "children",
                            "city", "close", "come", "company", "could", "country",
                            "course", "day", "different", "does", "done", "down",
                            "during", "each", "early", "either", "enough", "even",
                            "event", "ever", "every", "example", "family", "far",
                            "few", "first", "following", "form", "found", "general",
                            "given", "going", "good", "government", "great", "group",
                            "hand", "having", "head", "help", "here", "high", "home",
                            "house", "however", "human", "important", "include",
                            "information", "interest", "just", "keep", "kind", "know",
                            "knowledge", "large", "last", "later", "least", "left",
                            "less", "level", "life", "like", "likely", "line", "little",
                            "live", "local", "long", "look", "made", "make", "making",
                            "many", "might", "more", "most", "much", "must", "name",
                            "national", "near", "need", "never", "next", "night",
                            "nothing", "now", "number", "often", "once", "only", "open",
                            "order", "other", "others", "outside", "own", "part",
                            "people", "place", "point", "political", "possible",
                            "power", "present", "president", "problem", "program",
                            "provide", "public", "quite", "rather", "really", "report",
                            "research", "result", "right", "room", "said", "same",
                            "school", "second", "section", "service", "several",
                            "short", "should", "show", "side", "since", "small",
                            "social", "something", "sometimes", "soon", "state",
                            "still", "story", "study", "such", "system", "take",
                            "taken", "thing", "things", "think", "those", "thought",
                            "three", "through", "thus", "time", "together", "today",
                            "together", "took", "toward", "turn", "turned", "under",
                            "understand", "until", "upon", "used", "using", "usually",
                            "various", "very", "want", "water", "well", "went",
                            "where", "whether", "which", "while", "white", "whole",
                            "whose", "within", "without", "word", "work", "working",
                            "world", "would", "year", "years", "young",
                            "sequence", "position", "language", "attention", "model",
                            "network", "function", "process", "method", "approach",
                            "performance", "training", "learning", "machine",
                        })
                        if (frag.lower() in COMMON_PREFIXES
                                or frag.lower() in COMMON_WORDS
                                or len(frag) <= 3):
                            # Intentional compound hyphen — keep it, join lines
                            result[-1] = prev + cont
                            stripped = stripped[cont_match.end():].lstrip()
                            if stripped:
                                result.append(stripped)
                            continue
                        # Looks like a line-break hyphen — strip it
                        candidate = frag + cont
                        if candidate.isalpha() and candidate.islower():
                            result[-1] = prev[: m.start(1)] + candidate
                            stripped = stripped[cont_match.end():].lstrip()
                            if stripped:
                                result.append(stripped)
                            continue
            # Join with previous line if it doesn't end with sentence-ending punctuation
            if prev and prev[-1] not in ".!?\"'»" and not stripped[0].isupper():
                result[-1] = prev + " " + stripped
                continue
        result.append(stripped)
    return "\n".join(result)


def generate_audio(text: str, voice: str = "af_heart", speed: float = 1.0) -> np.ndarray | None:
    """Run Kokoro TTS on text, return concatenated audio as numpy array (24kHz float32)."""
    pipeline = _get_pipeline()
    text = normalize_text(text)
    chunks = list(pipeline(text, voice=voice, speed=speed, split_pattern=r"\n{2,}"))
    if not chunks:
        return None
    return np.concatenate([audio for _, _, audio in chunks])


def generate_audio_streaming(
    text: str,
    voice: str = "af_heart",
    speed: float = 1.0,
    *,
    progress_callback=None,
) -> np.ndarray | None:
    """
    Like generate_audio, but calls progress_callback(chunk_num, total_estimate)
    after each chunk. total_estimate is approximate.
    """
    pipeline = _get_pipeline()
    audio_chunks = []
    text = normalize_text(text)
    est_total = max(1, len(text) // 50)
    for i, (gs, ps, audio) in enumerate(pipeline(text, voice=voice, speed=speed, split_pattern=r"\n{2,}")):
        audio_chunks.append(audio)
        if progress_callback:
            progress_callback(i + 1, est_total)
    if not audio_chunks:
        return None
    return np.concatenate(audio_chunks)


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
