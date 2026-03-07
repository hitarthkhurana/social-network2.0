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

import cv2
import whisper
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import TEMP_DIR, FRAME_INTERVAL_SEC, CHUNK_DURATION_SEC
from processor import process_chunk
from storage import init_db, get_stats

console = Console()


def download_video(url: str) -> str:
    """Download YouTube video to temp dir, return path."""
    out_template = os.path.join(TEMP_DIR, "%(id)s.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[ext=mp4][height<=720]+bestaudio/best[ext=mp4]/best",
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

    # Find the downloaded file
    for f in Path(TEMP_DIR).glob("*.mp4"):
        return str(f)
    raise RuntimeError("Downloaded file not found")


def extract_audio(video_path: str) -> str:
    """Extract audio as WAV for transcription."""
    audio_path = video_path.replace(".mp4", ".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path]
    subprocess.run(cmd, capture_output=True)
    return audio_path


def transcribe_audio(audio_path: str) -> list[dict]:
    """
    Transcribe audio in 30-second windows to avoid Whisper hallucination.
    Whisper's context window is 30s — longer inputs cause repetition/hallucination.
    Returns list of {start, end, text} segments with correct global timestamps.
    """
    console.print("[cyan]Transcribing audio in 30s windows...[/cyan]")
    model = whisper.load_model("base")

    # Get total duration via ffprobe
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    total_duration = float(probe.stdout.strip() or 0)

    all_segments = []
    window = 30  # seconds — Whisper's max reliable context

    t = 0.0
    while t < total_duration:
        end = min(t + window, total_duration)

        # Extract 30s audio slice
        chunk_path = os.path.join(TEMP_DIR, f"audio_chunk_{int(t)}.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", audio_path,
             "-ss", str(t), "-t", str(window),
             "-ac", "1", "-ar", "16000", chunk_path],
            capture_output=True
        )

        # Transcribe this 30s window
        result = model.transcribe(chunk_path, verbose=False)

        # Offset timestamps to global position
        for seg in result.get("segments", []):
            all_segments.append({
                "start": seg["start"] + t,
                "end":   seg["end"] + t,
                "text":  seg["text"]
            })

        # Cleanup chunk
        try:
            os.remove(chunk_path)
        except Exception:
            pass

        t += window

    return all_segments


def get_transcript_for_window(segments: list[dict], start_sec: float, end_sec: float) -> str:
    """Get transcript text for a time window."""
    text = " ".join(
        seg["text"].strip()
        for seg in segments
        if seg["start"] >= start_sec and seg["end"] <= end_sec
    )
    return text.strip()


def extract_frames_for_window(video_path: str, start_sec: float, end_sec: float) -> list[str]:
    """Extract frames from a time window, return list of temp JPEG paths."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    frames = []
    t = start_sec
    while t < end_sec:
        frame_num = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            break
        path = os.path.join(TEMP_DIR, f"frame_{int(t*1000)}.jpg")
        cv2.imwrite(path, frame)
        frames.append(path)
        t += FRAME_INTERVAL_SEC

    cap.release()
    return frames


def cleanup_temp_frames(frames: list[str]):
    for f in frames:
        try:
            os.remove(f)
        except Exception:
            pass


def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return total_frames / fps


def process_video(url: str):
    """Full pipeline: download → transcribe → chunk → process → store."""
    init_db()

    # Download
    video_path = download_video(url)
    video_name = Path(video_path).stem
    console.print(f"[green]Downloaded:[/green] {video_path}")

    # Get duration
    duration = get_video_duration(video_path)
    total_chunks = int(duration / CHUNK_DURATION_SEC) + 1
    console.print(f"[cyan]Duration:[/cyan] {duration:.0f}s → {total_chunks} chunks of {CHUNK_DURATION_SEC}s")

    # Transcribe
    audio_path = extract_audio(video_path)
    segments = transcribe_audio(audio_path)

    # Process chunks
    console.print(f"\n[bold]Processing {total_chunks} chunks...[/bold]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console
    ) as progress:
        task = progress.add_task("Processing chunks", total=total_chunks)

        for i, start in enumerate(range(0, int(duration), CHUNK_DURATION_SEC)):
            end = min(start + CHUNK_DURATION_SEC, duration)
            timestamp = (datetime.utcnow() - timedelta(seconds=duration - start)).isoformat()

            progress.update(task, description=f"Chunk {i+1}/{total_chunks} [{start:.0f}s–{end:.0f}s]")

            frames = extract_frames_for_window(video_path, start, end)
            transcript = get_transcript_for_window(segments, start, end)

            if frames:
                process_chunk(
                    frames=frames,
                    transcript=transcript,
                    source=video_name,
                    timestamp=timestamp,
                    chunk_index=i
                )
                cleanup_temp_frames(frames)

            progress.advance(task)

    # Cleanup
    try:
        os.remove(video_path)
        os.remove(audio_path)
    except Exception:
        pass

    # Print stats
    stats = get_stats()
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Total memories: {stats['total_memories']}")
    console.print(f"  High importance: {stats['high_importance']}")
    console.print(f"  Low importance:  {stats['low_importance']}")
    console.print(f"  Avg importance:  {stats['avg_importance']}")
    console.print(f"  DB size:         {stats['db_size_kb']} KB")
    console.print(f"  Keyframes:       {stats['keyframes_size_kb']} KB")


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
        console.print("[red]No URLs provided.[/red]")
        console.print("Usage: python ingest.py <youtube_url>")
        sys.exit(1)

    for url in urls:
        console.rule(f"[bold]{url}[/bold]")
        process_video(url)


if __name__ == "__main__":
    main()
