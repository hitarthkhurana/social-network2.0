"""
record.py — Record a video from webcam and ingest it into memory.
Usage:
    python record.py              # records until you press Q
    python record.py --seconds 30 # records for 30 seconds
"""

import cv2
import subprocess
import argparse
import os
import time
from datetime import datetime
from rich.console import Console

from config import TEMP_DIR
from storage import init_db

console = Console()


def record(seconds: int = None) -> str:
    """Record from webcam, save as MP4, return path."""
    os.makedirs(TEMP_DIR, exist_ok=True)
    out_path = os.path.join(TEMP_DIR, f"webcam_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")

    # Try camera indices 0, 1, 2 in case default isn't available
    cap = None
    for idx in range(3):
        c = cv2.VideoCapture(idx)
        if c.isOpened():
            cap = c
            break
        c.release()

    if cap is None:
        raise RuntimeError(
            "Could not open webcam.\n"
            "On Mac: System Settings → Privacy & Security → Camera → enable for Terminal/iTerm2"
        )

    # Warmup — Continuity Camera (iPhone) needs a moment to start streaming
    console.print("[dim]Warming up camera...[/dim]")
    for _ in range(30):
        cap.read()
    time.sleep(1)

    fps    = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Write to a temp .avi first (OpenCV codec support is more reliable),
    # then convert to .mp4 with ffmpeg for Gemini compatibility.
    tmp_path = out_path.replace(".mp4", "_raw.avi")
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

    if seconds:
        console.print(f"[cyan]Recording for {seconds} seconds...[/cyan] (press Q to stop early)")
    else:
        console.print("[cyan]Recording...[/cyan] Press [bold]Q[/bold] to stop")

    start = time.time()
    frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        writer.write(frame)
        frames += 1

        # Show live preview
        elapsed = time.time() - start
        label = f"REC {elapsed:.1f}s | Press Q to stop"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Engram — Recording", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            break
        if seconds and elapsed >= seconds:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()

    duration = time.time() - start

    if frames == 0:
        raise RuntimeError(
            "Webcam opened but captured 0 frames.\n"
            "On Mac: System Settings → Privacy & Security → Camera → enable for Terminal/iTerm2"
        )

    # Convert AVI → MP4 for Gemini compatibility
    console.print("[cyan]Converting to MP4...[/cyan]")
    result = subprocess.run(
        ["ffmpeg", "-y", "-i", tmp_path, "-c:v", "libx264", "-preset", "fast", "-c:a", "aac", out_path],
        capture_output=True
    )
    os.remove(tmp_path)

    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg conversion failed: {result.stderr.decode()}")

    console.print(f"[green]Recorded {duration:.1f}s ({frames} frames) → {out_path}[/green]")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Record webcam video and ingest into memory")
    parser.add_argument("--seconds", type=int, default=None, help="Recording duration in seconds")
    parser.add_argument("--no-ingest", action="store_true", help="Save video without ingesting")
    args = parser.parse_args()

    # Record
    video_path = record(seconds=args.seconds)

    if args.no_ingest:
        console.print(f"[yellow]Saved to {video_path} (not ingested)[/yellow]")
        return

    # Ingest
    console.print(f"\n[bold]Ingesting into memory...[/bold]")
    from ingest import process_video
    init_db()
    process_video(video_path)


if __name__ == "__main__":
    main()
