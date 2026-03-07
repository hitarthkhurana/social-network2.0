"""
ingest.py — Download YouTube videos and process them into memories.
Usage:
    python ingest.py <youtube_url> [youtube_url2 ...]
    python ingest.py --file urls.txt
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import TEMP_DIR, CHUNK_DURATION_SEC
from processor import process_chunk
from storage import init_db, get_stats

console = Console()


def download_video(url: str) -> str:
    """Download YouTube video to temp dir, return path."""
    out_template = os.path.join(TEMP_DIR, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "--merge-output-format", "mp4",
        "-o", out_template,
        "--no-playlist",
        url
    ]
    console.print(f"[cyan]Downloading:[/cyan] {url}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"[red]Download failed:[/red] {result.stderr}")
        raise RuntimeError(f"yt-dlp failed: {result.stderr}")

    for f in sorted(Path(TEMP_DIR).glob("*.mp4"), key=os.path.getmtime, reverse=True):
        return str(f)
    raise RuntimeError("Downloaded file not found")


def get_video_duration(video_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True
    )
    return float(result.stdout.strip() or 0)


def split_into_chunks(video_path: str, duration: float) -> list[str]:
    """Split video into 60s chunks using ffmpeg. Returns list of chunk paths."""
    chunks = []
    i = 0
    t = 0.0

    while t < duration:
        chunk_path = os.path.join(TEMP_DIR, f"chunk_{i:04d}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(t),
            "-i", video_path,
            "-t", str(CHUNK_DURATION_SEC),
            "-c", "copy",          # fast — no re-encoding
            chunk_path
        ]
        subprocess.run(cmd, capture_output=True)
        if os.path.exists(chunk_path):
            chunks.append(chunk_path)
        t += CHUNK_DURATION_SEC
        i += 1

    return chunks


def process_video(url: str):
    """Full pipeline: download (or use local file) → split → analyze with Gemini → store."""
    init_db()

    # Use local file directly if path exists, otherwise download from YouTube
    if os.path.exists(url):
        video_path = url
        console.print(f"[green]Using local file:[/green] {video_path}")
    else:
        video_path = download_video(url)
    video_name = Path(video_path).stem
    console.print(f"[green]Downloaded:[/green] {video_path}")

    # Get duration
    duration = get_video_duration(video_path)
    total_chunks = int(duration / CHUNK_DURATION_SEC) + 1
    console.print(f"[cyan]Duration:[/cyan] {duration:.0f}s → {total_chunks} chunks of {CHUNK_DURATION_SEC}s")

    # Split into chunks (no Whisper — Gemini handles audio natively)
    console.print("[cyan]Splitting into chunks...[/cyan]")
    chunks = split_into_chunks(video_path, duration)

    # Process each chunk
    console.print(f"\n[bold]Processing {len(chunks)} chunks with Gemini...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing", total=len(chunks))

        for i, chunk_path in enumerate(chunks):
            start_sec = i * CHUNK_DURATION_SEC
            timestamp = (datetime.utcnow() - timedelta(seconds=duration - start_sec)).isoformat()

            progress.update(task, description=f"Chunk {i+1}/{len(chunks)} [{start_sec:.0f}s–{start_sec+CHUNK_DURATION_SEC:.0f}s]")

            process_chunk(
                chunk_path=chunk_path,
                source=video_name,
                timestamp=timestamp,
                chunk_index=i
            )

            # Delete chunk after processing
            try:
                os.remove(chunk_path)
            except Exception:
                pass

            progress.advance(task)

    # Delete original video
    try:
        os.remove(video_path)
    except Exception:
        pass

    # Stats
    stats = get_stats()
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Total memories : {stats['total_memories']}")
    console.print(f"  High importance: {stats['high_importance']}")
    console.print(f"  Low importance : {stats['low_importance']}")
    console.print(f"  Avg importance : {stats['avg_importance']}")
    console.print(f"  DB size        : {stats['db_size_kb']} KB")
    console.print(f"  Clips saved    : {stats['clips_size_kb']} KB")


def main():
    parser = argparse.ArgumentParser(description="Ingest YouTube videos into memory system")
    parser.add_argument("urls", nargs="*", help="YouTube URLs to process")
    parser.add_argument("--file", help="Text file with one URL per line")
    args = parser.parse_args()

    urls = list(args.urls)
    if args.file:
        with open(args.file) as f:
            urls += [line.strip() for line in f if line.strip()]

    if not urls:
        console.print("[red]No input provided.[/red]")
        console.print("Usage: python ingest.py <youtube_url or /path/to/video.mp4>")
        sys.exit(1)

    for url in urls:
        console.rule(f"[bold]{url}[/bold]")
        process_video(url)


if __name__ == "__main__":
    main()
