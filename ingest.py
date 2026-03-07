"""
ingest.py — Process local video files into person-indexed memories.
Usage:
    python ingest.py <video.mp4> [video2.mp4 ...]
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timedelta

import cv2
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from config import TEMP_DIR, FRAME_INTERVAL_SEC, CHUNK_DURATION_SEC, GEMINI_API_KEY
from processor import process_chunk
from storage import init_db, get_stats
from faces import init_face_app
from google import genai
from google.genai import types

console = Console()


def extract_audio(video_path: str) -> str:
    audio_path = os.path.join(TEMP_DIR, Path(video_path).stem + ".wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-ac", "1", "-ar", "16000", audio_path]
    subprocess.run(cmd, capture_output=True)
    return audio_path


def transcribe_audio(audio_path: str) -> str:
    """Transcribe full audio using Gemini."""
    console.print("[cyan]Transcribing audio with Gemini...[/cyan]")
    client = genai.Client(api_key=GEMINI_API_KEY)

    with open(audio_path, "rb") as f:
        audio_data = f.read()

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=types.Content(
            parts=[
                types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_data)),
                types.Part(text=(
                    "Transcribe all speech in this audio. Return ONLY the transcription text, "
                    "nothing else. If there is no speech, return 'No speech detected.'"
                )),
            ]
        ),
    )
    transcript = response.text.strip()
    console.print(f"[green]Transcript:[/green] {transcript[:100]}{'...' if len(transcript) > 100 else ''}")
    return transcript


def extract_frames_for_window(video_path: str, start_sec: float, end_sec: float) -> list:
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


def cleanup_temp_frames(frames: list):
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


def process_video(video_path: str):
    if not os.path.exists(video_path):
        console.print(f"[red]File not found:[/red] {video_path}")
        return

    init_db()

    console.print("[cyan]Loading face recognition model...[/cyan]")
    face_app = init_face_app()

    video_name = Path(video_path).stem
    console.print(f"[green]Processing:[/green] {video_path}")

    duration = get_video_duration(video_path)
    total_chunks = int(duration / CHUNK_DURATION_SEC) + 1
    console.print(f"[cyan]Duration:[/cyan] {duration:.0f}s → {total_chunks} chunks of {CHUNK_DURATION_SEC}s")

    audio_path = extract_audio(video_path)
    transcript = transcribe_audio(audio_path)

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

            if frames:
                process_chunk(
                    frames=frames,
                    transcript=transcript,
                    source=video_name,
                    timestamp=timestamp,
                    chunk_index=i,
                    face_app=face_app,
                )
                cleanup_temp_frames(frames)

            progress.advance(task)

    try:
        os.remove(audio_path)
    except Exception:
        pass

    stats = get_stats()
    console.print(f"\n[bold green]Done![/bold green]")
    console.print(f"  Total memories: {stats['total_memories']}")
    console.print(f"  High importance: {stats['high_importance']}")
    console.print(f"  Low importance:  {stats['low_importance']}")
    console.print(f"  Avg importance:  {stats['avg_importance']}")
    console.print(f"  Persons:         {stats['persons']}")
    console.print(f"  DB size:         {stats['db_size_kb']} KB")
    console.print(f"  Keyframes:       {stats['keyframes_size_kb']} KB")


def main():
    parser = argparse.ArgumentParser(description="Process local videos into person-indexed memories")
    parser.add_argument("videos", nargs="+", help="Path(s) to video files")
    args = parser.parse_args()

    for video_path in args.videos:
        console.rule(f"[bold]{video_path}[/bold]")
        process_video(video_path)


if __name__ == "__main__":
    main()
