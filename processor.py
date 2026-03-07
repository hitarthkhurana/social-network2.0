import os
import json
import time
import subprocess
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import MultiModalEmbeddingModel, Image as VertexImage

from config import (
    GEMINI_FLASH_MODEL, EMBEDDING_MODEL,
    VERTEX_PROJECT, VERTEX_LOCATION,
    IMPORTANCE_HIGH, CLIPS_DIR, CHUNK_DURATION_SEC
)
from storage import save_memory, save_person, update_person, get_all_people
from faces import detect_faces, match_face

# All AI via Vertex AI — uses GCP project + $300 credits
vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
gemini_model = GenerativeModel(GEMINI_FLASH_MODEL)
embed_model  = MultiModalEmbeddingModel.from_pretrained(EMBEDDING_MODEL)


def call_with_retry(fn, retries=3, base_delay=15):
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            if "429" in msg or "RESOURCE_EXHAUSTED" in msg or "quota" in msg.lower():
                wait = base_delay * (attempt + 1)
                print(f"  [rate limit] waiting {wait}s before retry {attempt+1}/{retries}...")
                time.sleep(wait)
            else:
                raise
    return None


def analyze_chunk(chunk_path: str, source: str, timestamp: str) -> dict:
    """
    Single Gemini call via Vertex AI on the video chunk.
    Gemini handles audio + video natively — no Whisper needed.
    Returns: importance, summary, metadata, and exact timestamps of important segments.
    """
    print(f"  Sending chunk to Gemini via Vertex AI...")

    with open(chunk_path, "rb") as f:
        video_bytes = f.read()

    video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")

    prompt = """Analyze this video segment. Do two things:

1. Score overall importance 0.0-1.0:
   - People speaking, learning, discussion, active work → 0.5-1.0
   - Static visuals, silence, background, repetitive → 0.0-0.4

2. Find the exact timestamps of the most important moments within this clip.
   Only include segments where something meaningful is happening.
   Each segment should be 3-30 seconds long.

Return ONLY valid JSON:
{
  "importance": 0.0,
  "summary": "2-3 sentence description of the whole chunk",
  "people": ["people visible or speaking"],
  "activity": "main activity in one phrase",
  "tags": ["3-5 relevant tags"],
  "important_segments": [
    {"start": 0.0, "end": 0.0, "reason": "why this moment matters"}
  ]
}

If importance < 0.5, return empty important_segments array."""

    def _call():
        return gemini_model.generate_content([video_part, prompt])

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
        "summary": f"Video segment from {source} at {timestamp}",
        "people": [],
        "activity": "",
        "tags": [],
        "important_segments": []
    }


def cut_clips(video_path: str, segments: list[dict], memory_id_hint: str) -> list[str]:
    """Use ffmpeg to cut exact clips at the timestamps Gemini returned."""
    os.makedirs(CLIPS_DIR, exist_ok=True)
    clip_paths = []

    for i, seg in enumerate(segments):
        start = seg.get("start", 0)
        end   = seg.get("end", 0)
        duration = end - start
        if duration <= 0:
            continue

        out_path = os.path.join(CLIPS_DIR, f"{memory_id_hint}_{i}.mp4")
        cmd = [
            "ffmpeg", "-y",
            "-ss", str(start),
            "-i", video_path,
            "-t", str(duration),
            "-c:v", "libx264", "-c:a", "aac",
            "-preset", "fast",
            out_path
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode == 0:
            clip_paths.append(out_path)
            print(f"  Clip saved: {start:.1f}s–{end:.1f}s ({duration:.1f}s) → {os.path.basename(out_path)}")

    return clip_paths


def extract_keyframe(chunk_path: str) -> str | None:
    """Extract a single frame from the middle of the chunk using ffmpeg."""
    keyframe_path = chunk_path.replace(".mp4", "_keyframe.jpg")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(CHUNK_DURATION_SEC // 2),  # grab from midpoint
        "-i", chunk_path,
        "-vframes", "1",
        "-q:v", "2",
        keyframe_path
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode == 0 and os.path.exists(keyframe_path):
        return keyframe_path
    return None


def get_embedding(text: str, image_path: str | None = None) -> np.ndarray:
    """
    True multimodal embedding via Vertex AI multimodalembedding@001.
    Embeds image + text into a shared 1408-dim vector space.
    Falls back to text-only if no image provided.
    """
    try:
        if image_path and os.path.exists(image_path):
            image = VertexImage.load_from_file(image_path)
            result = embed_model.get_embeddings(
                image=image,
                contextual_text=text
            )
            # Use image embedding — it already incorporates text context
            embedding = result.image_embedding
            print(f"  Embedding: multimodal 1408-dim (image + text)")
        else:
            result = embed_model.get_embeddings(contextual_text=text)
            embedding = result.text_embedding
            print(f"  Embedding: text-only 1408-dim")

        return np.array(embedding, dtype=np.float32)

    except Exception as e:
        print(f"  [embedding error] {e}")
        return np.zeros(1408, dtype=np.float32)


def process_chunk(chunk_path: str, source: str, timestamp: str, chunk_index: int):
    """
    Full pipeline for one 60s chunk:
    1. Send video to Gemini → get metadata + important timestamps
    2. Cut clips at those timestamps with ffmpeg
    3. Embed summary
    4. Save memory
    """
    print(f"\n  Chunk {chunk_index}: {os.path.basename(chunk_path)}")

    # Step 1: Gemini analyzes video natively (no Whisper)
    result = analyze_chunk(chunk_path, source, timestamp)
    importance = float(result.get("importance", 0.5))
    detail_level = "high" if importance >= IMPORTANCE_HIGH else "low"
    segments = result.get("important_segments", [])

    print(f"  Importance: {importance:.2f} — {result.get('activity', '')}")
    print(f"  Summary: {result['summary'][:80]}...")
    print(f"  Important segments found: {len(segments)}")

    # Step 2: Cut exact clips with ffmpeg
    memory_id_hint = f"{source}_{chunk_index:04d}"
    clip_paths = cut_clips(chunk_path, segments, memory_id_hint) if segments else []

    # Step 3: Detect + store faces from keyframe
    keyframe_path = extract_keyframe(chunk_path)
    memory_id_hint_faces = f"{source}_{chunk_index:04d}"
    _process_faces(keyframe_path, memory_id_hint_faces, timestamp)

    # Step 4: Multimodal embed (image + text)
    if keyframe_path:
        print(f"  Embedding: multimodal (image + text)")
    else:
        print(f"  Embedding: text only (no keyframe)")
    embedding = get_embedding(result["summary"], image_path=keyframe_path)

    # Cleanup keyframe after embedding
    if keyframe_path and os.path.exists(keyframe_path):
        os.remove(keyframe_path)

    # Step 4: Save memory
    memory = {
        "timestamp": timestamp,
        "source": source,
        "summary": result["summary"],
        "detail_level": detail_level,
        "people": result.get("people", []),
        "activity": result.get("activity", ""),
        "tags": result.get("tags", []),
        "importance": importance,
        "clip_paths": clip_paths,
        "segments": segments,
        "embedding": embedding,
    }

    saved = save_memory(memory)

    # Link faces to this memory now that we have the memory ID
    _link_faces_to_memory(source, chunk_index, saved)

    return memory


def _process_faces(keyframe_path: str | None, memory_id_hint: str, timestamp: str):
    """Detect faces in keyframe, match against known people or create new entry."""
    if not keyframe_path or not os.path.exists(keyframe_path):
        return

    from faces import detect_faces, match_face
    faces = detect_faces(keyframe_path)
    if not faces:
        return

    print(f"  Faces detected: {len(faces)}")
    known_people = get_all_people()

    for face in faces:
        match = match_face(face["embedding"], known_people, threshold=0.5)
        if match:
            print(f"  Known person: {match.get('name', 'unknown')} (score: {match['match_score']:.2f})")
            update_person(match["id"], last_seen=timestamp)
        else:
            # New person — store with unknown name for now
            name = f"unknown_{len(known_people) + 1}"
            save_person({
                "name": name,
                "face_embedding": face["embedding"],
                "face_crop_path": face.get("crop_path", ""),
                "first_seen": timestamp,
                "last_seen": timestamp,
                "memory_ids": []
            })
            print(f"  New person saved as: {name}")
            known_people = get_all_people()  # refresh


def _link_faces_to_memory(source: str, chunk_index: int, memory_id: int | None):
    """Link the memory ID back to people seen in this chunk."""
    if not memory_id:
        return
    people = get_all_people()
    for person in people:
        if memory_id not in person["memory_ids"]:
            update_person(person["id"], memory_id=memory_id)
