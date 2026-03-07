import os
import json
import time
from datetime import datetime
import numpy as np
from PIL import Image
from google import genai
from google.genai import types

from config import (
    GEMINI_API_KEY, GEMINI_FLASH_MODEL, EMBEDDING_MODEL,
    IMPORTANCE_HIGH, IMPORTANCE_LOW, KEYFRAMES_BY_IMPORTANCE,
    KEYFRAMES_DIR, MAX_FRAMES_TO_GEMINI
)
from storage import save_memory

client = genai.Client(api_key=GEMINI_API_KEY)


def call_with_retry(fn, retries=3, base_delay=30):
    """Call fn(), retrying on 429 rate limit errors with backoff."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
                wait = base_delay * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    return None


def get_keyframe_count(importance: float) -> int:
    for (low, high), count in KEYFRAMES_BY_IMPORTANCE.items():
        if low <= importance <= high:
            return count
    return 0


def read_image_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def analyze_chunk(frames: list[str], transcript: str, source: str, timestamp: str) -> dict:
    """
    Single API call that scores importance AND summarizes in one shot.
    Returns full memory dict with importance + summary fields.
    """
    parts = []
    for frame_path in frames[:MAX_FRAMES_TO_GEMINI]:
        parts.append(types.Part.from_bytes(
            data=read_image_bytes(frame_path),
            mime_type="image/jpeg"
        ))

    prompt = f"""Analyze this video segment. Score its importance AND summarize it in one response.

Transcript: "{transcript[:800]}"
Source: {source} | Time: {timestamp}

Importance scoring rules:
- People talking, learning, discussion, active work → score 0.5-1.0
- Static visuals, no speech, background, repetitive → score 0.0-0.4

If importance >= 0.5 → provide full detail (summary, people, activity, tags)
If importance < 0.5  → provide only a one-line summary and 1-2 tags

Return ONLY valid JSON:
{{
  "importance": 0.0,
  "reason": "one line explanation",
  "summary": "description of what is happening",
  "people": ["only if importance >= 0.5, else empty array"],
  "activity": "only if importance >= 0.5, else empty string",
  "tags": ["1-5 tags"]
}}"""

    parts.append(prompt)

    def _call():
        return client.models.generate_content(model=GEMINI_FLASH_MODEL, contents=parts)

    try:
        response = call_with_retry(_call)
        if response is None:
            return _fallback(source, timestamp)
        text = response.text.strip()
        if "```" in text:
            text = text.split("```")[1].replace("json", "").strip()
        return json.loads(text)
    except Exception as e:
        print(f"  [analyze error] {e}")
        return _fallback(source, timestamp)


def _fallback(source: str, timestamp: str) -> dict:
    return {
        "importance": 0.5,
        "reason": "fallback",
        "summary": f"Video segment from {source} at {timestamp}",
        "people": [],
        "activity": "",
        "tags": []
    }


def get_embedding(text: str) -> np.ndarray:
    def _call():
        return client.models.embed_content(model=EMBEDDING_MODEL, contents=text)

    try:
        result = call_with_retry(_call)
        if result is None:
            return np.zeros(768, dtype=np.float32)
        return np.array(result.embeddings[0].values, dtype=np.float32)
    except Exception as e:
        print(f"  [embedding error] {e}")
        return np.zeros(768, dtype=np.float32)


def save_keyframes(frames: list[str], memory_id_hint: str, count: int) -> list[str]:
    if count == 0 or not frames:
        return []

    indices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    selected = [frames[i] for i in indices]

    saved_paths = []
    for i, frame_path in enumerate(selected):
        img = Image.open(frame_path)
        img.thumbnail((320, 240))
        out_path = os.path.join(KEYFRAMES_DIR, f"{memory_id_hint}_{i}.jpg")
        img.save(out_path, "JPEG", quality=70)
        saved_paths.append(out_path)

    return saved_paths


def process_chunk(frames: list[str], transcript: str, source: str, timestamp: str, chunk_index: int):
    """Process one chunk: analyze (score+summarize) → embed → store."""
    print(f"\n  Chunk {chunk_index}: {len(frames)} frames, transcript: {len(transcript)} chars")

    # Step 1: Single call — score + summarize
    result = analyze_chunk(frames, transcript, source, timestamp)
    importance = float(result.get("importance", 0.5))
    detail_level = "high" if importance >= IMPORTANCE_HIGH else "low"

    print(f"  Importance: {importance:.2f} — {result.get('reason', '')}")
    print(f"  Summary [{detail_level}]: {result['summary'][:80]}...")

    # Step 2: Save keyframes
    keyframe_count = get_keyframe_count(importance)
    memory_id_hint = f"{source}_{chunk_index:04d}"
    keyframe_paths = save_keyframes(frames, memory_id_hint, keyframe_count)
    print(f"  Keyframes stored: {len(keyframe_paths)}")

    # Step 3: Embed the summary
    embedding = get_embedding(result["summary"])

    # Step 4: Save memory
    memory = {
        "timestamp": timestamp,
        "source": source,
        "summary": result["summary"],
        "detail_level": detail_level,
        "people": result.get("people", []),
        "activity": result.get("activity", ""),
        "tags": result.get("tags", []),
        "transcript": transcript if importance >= IMPORTANCE_HIGH else None,
        "importance": importance,
        "keyframe_paths": keyframe_paths,
        "embedding": embedding,
    }

    save_memory(memory)
    return memory
