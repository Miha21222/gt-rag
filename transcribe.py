"""Transcribe GT conference recordings (Google Drive) with faster-whisper.

Processes one recording at a time to cap peak disk use: download video ->
transcribe -> write markdown transcript -> delete video. Resumable: skips
sources whose transcript file already exists.

Usage:
    python transcribe.py            # all pending
    python transcribe.py --only conf-b1-market-mechanics
    python transcribe.py --model medium --device cpu
"""
from __future__ import annotations

import argparse
import os
import re
import site
import sys
import time
from pathlib import Path

# Make pip-installed NVIDIA runtime DLLs (cuBLAS / cuDNN) visible to
# CTranslate2 before it loads. Harmless if absent (CPU fallback).
for _sp in site.getsitepackages():
    for _sub in (r"nvidia\cublas\bin", r"nvidia\cudnn\bin"):
        _p = os.path.join(_sp, _sub)
        if os.path.isdir(_p):
            os.add_dll_directory(_p)
            os.environ["PATH"] = _p + os.pathsep + os.environ.get("PATH", "")

from gt_rag.common import RAW_DIR, ROOT, parse_front_matter

TMP_DIR = ROOT / "data" / "tmp"
DRIVE_RE = re.compile(r"https://drive\.google\.com/file/d/([\w-]+)")


def find_sources() -> list[dict]:
    """Conference/QA pages in data/raw that contain a Drive link and have no
    transcript yet."""
    sources = []
    for path in sorted(RAW_DIR.glob("*.md")):
        if path.stem.endswith("-transcript"):
            continue
        meta, body = parse_front_matter(path.read_text(encoding="utf-8"))
        if meta.get("type") not in ("conference", "qa"):
            continue
        m = DRIVE_RE.search(body)
        if not m:
            print(f"  skip {path.stem}: no Drive link")
            continue
        out = RAW_DIR / f"{path.stem}-transcript.md"
        if out.exists():
            print(f"  skip {path.stem}: transcript exists")
            continue
        sources.append({
            "slug": path.stem,
            "title": meta.get("title", path.stem),
            "block": meta.get("block"),
            "file_id": m.group(1),
            "out": out,
        })
    return sources


DOWNLOAD_TIMEOUT_S = 900
DOWNLOAD_ATTEMPTS = 3


def download(file_id: str, dest_dir: Path) -> Path | None:
    """Download in a subprocess so a hung connection (gdown has no socket
    timeout) can be killed and retried instead of stalling the whole run."""
    import subprocess

    dest_dir.mkdir(parents=True, exist_ok=True)
    code = (
        "import gdown, sys; "
        f"out = gdown.download(id={file_id!r}, output={str(dest_dir) + os.sep!r}, quiet=True); "
        "print(out if out else '', end='')"
    )
    for attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        # clear partial files from a killed previous attempt
        for leftover in dest_dir.iterdir():
            leftover.unlink(missing_ok=True)
        try:
            r = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=DOWNLOAD_TIMEOUT_S,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            out = r.stdout.strip()
            if r.returncode == 0:
                if out and Path(out).exists():
                    return Path(out)
                # stdout path can round-trip badly on Windows; the file gdown
                # left in the (previously emptied) dest_dir is the download
                files = [p for p in dest_dir.iterdir() if p.is_file()]
                if files:
                    return max(files, key=lambda p: p.stat().st_size)
            err = (r.stderr or "").strip().splitlines()
            print(f"  attempt {attempt} failed: {err[-1] if err else 'no output'}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  attempt {attempt} timed out after {DOWNLOAD_TIMEOUT_S}s", flush=True)
        time.sleep(30 * attempt)
    return None


def fmt_ts(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def transcribe_file(model, video: Path) -> list[tuple[float, str]]:
    """Return [(start_seconds, paragraph_text)] grouped into ~60s paragraphs."""
    segments, info = model.transcribe(
        str(video), language="ru", vad_filter=True, beam_size=5
    )
    paras: list[tuple[float, str]] = []
    buf: list[str] = []
    para_start = 0.0
    last_end = 0.0
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        gap = seg.start - last_end
        span = seg.end - para_start
        if buf and (gap > 2.5 or span > 60):
            paras.append((para_start, " ".join(buf)))
            buf = []
            para_start = seg.start
        if not buf:
            para_start = seg.start
        buf.append(text)
        last_end = seg.end
    if buf:
        paras.append((para_start, " ".join(buf)))
    return paras


def write_transcript(src: dict, paras: list[tuple[float, str]], model_name: str) -> None:
    block = src["block"] if src["block"] is not None else "null"
    lines = [
        "---",
        f'title: "{src["title"]} — транскрипт"',
        f'notion_url: ""',
        f"block: {block}",
        'type: "transcript"',
        f'slug: "{src["slug"]}-transcript"',
        f'crawled_at: "{time.strftime("%Y-%m-%d")}"',
        f'whisper_model: "{model_name}"',
        f'source_video: "https://drive.google.com/file/d/{src["file_id"]}/view"',
        "---",
        "",
    ]
    for start, text in paras:
        lines.append(f"### [{fmt_ts(start)}]")
        lines.append("")
        lines.append(text)
        lines.append("")
    src["out"].write_text("\n".join(lines), encoding="utf-8")


def load_model(model_name: str, device: str):
    from faster_whisper import WhisperModel

    if device in ("cuda", "auto"):
        try:
            m = WhisperModel(model_name, device="cuda", compute_type="int8_float16")
            print(f"model {model_name} on cuda (int8_float16)")
            return m, model_name
        except Exception as e:
            print(f"cuda unavailable ({e}); falling back to cpu")
    m = WhisperModel(model_name, device="cpu", compute_type="int8")
    print(f"model {model_name} on cpu (int8)")
    return m, model_name


def keep_awake() -> None:
    """Prevent Windows sleep while this process runs (display may still off).
    Automatically released when the process exits."""
    if sys.platform == "win32":
        import ctypes

        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )


def main() -> int:
    keep_awake()
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="large-v3")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--only", default=None, help="process a single slug")
    args = ap.parse_args()

    sources = find_sources()
    if args.only:
        sources = [s for s in sources if s["slug"] == args.only]
    if not sources:
        print("nothing to do")
        return 0
    print(f"{len(sources)} recordings to transcribe")

    model, model_name = load_model(args.model, args.device)

    ok, failed = [], []
    for i, src in enumerate(sources, 1):
        t0 = time.time()
        print(f"[{i}/{len(sources)}] {src['slug']}: downloading...", flush=True)
        video = None
        try:
            video = download(src["file_id"], TMP_DIR)
            if not video or not video.exists():
                raise RuntimeError("download failed (file may not be link-shared)")
            size_mb = video.stat().st_size / 1e6
            print(f"  downloaded {video.name} ({size_mb:.0f} MB), transcribing...", flush=True)
            paras = transcribe_file(model, video)
            if not paras:
                raise RuntimeError("empty transcript")
            write_transcript(src, paras, model_name)
            words = sum(len(t.split()) for _, t in paras)
            print(f"  done: {len(paras)} paragraphs, ~{words} words, "
                  f"{(time.time() - t0) / 60:.1f} min", flush=True)
            ok.append(src["slug"])
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            failed.append((src["slug"], str(e)))
        finally:
            if video and video.exists():
                video.unlink()

    print(f"\nsummary: {len(ok)} ok, {len(failed)} failed")
    for slug, err in failed:
        print(f"  {slug}: {err}")
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
