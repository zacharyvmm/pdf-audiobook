"""
End-to-end pipeline spike: PDF → chapters → TTS → MP3
"""
import fitz
from kokoro import KPipeline
import soundfile as sf
import subprocess
import re
import os
from pathlib import Path

OUT_DIR = Path(__file__).parent / "output"
OUT_DIR.mkdir(exist_ok=True)


def extract_chapters(pdf_path: str) -> list[tuple[str, str]]:
    """
    Extract text from PDF, split by chapter headings.
    Returns list of (chapter_title, chapter_text).
    For scientific articles: returns one chapter = whole article.
    """
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text() + "\n"

    # Try to split by chapter pattern
    chapter_pattern = re.compile(
        r'(?:^|\n)((?:Chapter|CHAPTER|Section)\s+\d+[.:\s].*?)(?=\n(?:Chapter|CHAPTER|Section)\s+\d+|$)',
        re.MULTILINE | re.DOTALL,
    )
    matches = list(chapter_pattern.finditer(full_text))

    if len(matches) >= 2:
        # Book with chapters
        chapters = []
        for i, m in enumerate(matches):
            title = m.group(1).strip()
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            body = full_text[start:end].strip()
            chapters.append((title, body))
        return chapters
    else:
        # Article or single block — use whole text
        clean = full_text.strip()
        if not clean:
            return []
        title = doc.metadata.get("title", "Untitled")
        if not title:
            title = Path(pdf_path).stem.replace("_", " ").title()
        return [(title, clean)]


def generate_audio(text: str, voice: str = "af_heart") -> bytes:
    """
    Run Kokoro TTS on text, return concatenated WAV bytes (24kHz mono float32).
    Returns raw PCM bytes or None.
    """
    pipeline = KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")
    chunks = list(pipeline(text, voice=voice, speed=1))
    if not chunks:
        return None

    # Concatenate all audio chunks
    import numpy as np

    all_audio = np.concatenate([audio for _, _, audio in chunks])
    return all_audio


def wav_to_mp3(wav_path: str, mp3_path: str):
    """Convert WAV to MP3 using ffmpeg."""
    subprocess.run(
        [
            "ffmpeg", "-y", "-i", wav_path,
            "-codec:a", "libmp3lame", "-qscale:a", "2",
            mp3_path,
        ],
        capture_output=True,
        check=True,
    )


def main():
    import os
    base = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base, "../002-pdf-extraction/samples/test_book.pdf")
    print(f"Processing: {pdf_path}")

    # Step 1: Extract chapters
    chapters = extract_chapters(pdf_path)
    print(f"Found {len(chapters)} chapter(s)")

    mp3_files = []
    for i, (title, text) in enumerate(chapters):
        safe_title = re.sub(r"[^\w\s-]", "", title)[:50].strip()
        print(f"\n  Chapter {i+1}: {safe_title}")
        print(f"  Text: {len(text)} chars, ~{len(text.split())} words")

        # Step 2: TTS
        print(f"  Generating audio...")
        audio = generate_audio(text)
        if audio is None:
            print(f"  SKIP: no audio generated")
            continue

        # Step 3: Save WAV
        import soundfile as sf

        wav_path = OUT_DIR / f"ch_{i+1:02d}_{safe_title}.wav"
        sf.write(str(wav_path), audio, 24000)
        dur = len(audio) / 24000
        print(f"  WAV: {wav_path.name} ({dur:.1f}s)")

        # Step 4: Convert to MP3
        mp3_path = OUT_DIR / f"ch_{i+1:02d}_{safe_title}.mp3"
        wav_to_mp3(str(wav_path), str(mp3_path))
        size_kb = mp3_path.stat().st_size / 1024
        print(f"  MP3: {mp3_path.name} ({size_kb:.0f} KB)")
        mp3_files.append(mp3_path)

    print(f"\nDone! {len(mp3_files)} MP3 files in {OUT_DIR}")
    for f in sorted(OUT_DIR.glob("*.mp3")):
        print(f"  {f.name} ({f.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
