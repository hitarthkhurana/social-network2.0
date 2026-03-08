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
from storage import save_memory, save_person, update_person, link_person_memory, get_all_people
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

    prompt = """You are analyzing a video segment for a personal memory system. Be thorough.

1. SCENE DESCRIPTION — describe everything visible:
   - Where is this? (room type, setting, environment)
   - What objects, screens, documents are visible?
   - What are people doing physically (typing, gesturing, reading)?
   - What is on any visible screens or whiteboards?

2. SPEECH TRANSCRIPT — transcribe every word spoken verbatim.
   Include speaker labels if multiple people (e.g. "Person A: ..., Person B: ...").
   If silent, use empty string.

3. IMPORTANCE SCORE 0.0-1.0:
   - Active conversation, learning, work, decisions → 0.6-1.0
   - Someone speaking alone to camera → 0.4-0.6
   - Idle, browsing, silence, nothing happening → 0.0-0.3

4. IMPORTANT TIMESTAMPS — start/end seconds of meaningful moments.
   Leave empty if importance < 0.5.

Return ONLY valid JSON — no markdown, no extra text:
{
  "importance": 0.0,
  "summary": "detailed 3-5 sentence description of the scene and what happened",
  "scene": "precise description of the physical environment and visible content",
  "transcript": "verbatim speech with speaker labels, or empty string if silent",
  "people": ["use actual name if mentioned in speech, otherwise describe appearance"],
  "activity": "main activity in one phrase",
  "tags": ["4-6 relevant tags"],
  "important_segments": [
    {"start": 0.0, "end": 0.0, "reason": "why this moment matters"}
  ]
}"""

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

    # Get actual duration so we don't seek past the end of short clips
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", chunk_path],
        capture_output=True, text=True
    )
    try:
        duration = float(probe.stdout.strip())
    except (ValueError, TypeError):
        duration = CHUNK_DURATION_SEC
    seek = min(CHUNK_DURATION_SEC // 2, max(0, duration / 2))

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(seek),
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

    # Step 3: Detect + store faces from keyframe, using Gemini's people list for naming
    keyframe_path = extract_keyframe(chunk_path)
    seen_persons = _process_faces(
        keyframe_path, f"{source}_{chunk_index:04d}", timestamp,
        gemini_people=result.get("people", [])
    )

    # Step 4: Multimodal embed — combine summary + transcript + scene for richer search
    embed_text = result["summary"]
    if result.get("transcript"):
        embed_text += " " + result["transcript"]
    if result.get("scene"):
        embed_text += " " + result["scene"]
    if keyframe_path:
        print(f"  Embedding: multimodal (image + text + transcript)")
    else:
        print(f"  Embedding: text only (summary + transcript)")
    embedding = get_embedding(embed_text, image_path=keyframe_path)

    # Cleanup keyframe after embedding
    if keyframe_path and os.path.exists(keyframe_path):
        os.remove(keyframe_path)

    # Step 5: Save memory
    memory = {
        "timestamp": timestamp,
        "source": source,
        "summary": result["summary"],
        "scene": result.get("scene", ""),
        "transcript": result.get("transcript", ""),
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

    # Link only the faces seen in this chunk to this memory (with confidence scores)
    _link_faces_to_memory(seen_persons, saved)

    return memory


def _process_faces(keyframe_path: str | None, memory_id_hint: str, timestamp: str,
                   gemini_people: list[str] = None) -> list[tuple]:
    """Detect faces in keyframe, match against known people or create new entry.
    Uses Gemini's people list to auto-name unknown faces.
    Returns list of (person_id, confidence) tuples."""
    if not keyframe_path or not os.path.exists(keyframe_path):
        return []

    from faces import detect_faces, match_face
    faces = detect_faces(keyframe_path)
    if not faces:
        return []

    print(f"  Faces detected: {len(faces)}")
    known_people = get_all_people()
    seen_persons = []  # list of (person_id, confidence)

    # Names Gemini identified — split into real names vs descriptions
    all_gemini = [n for n in (gemini_people or []) if n]
    # A "real name" is short (1-3 words) and not a physical description
    description_keywords = {"man", "woman", "person", "student", "young", "old", "wearing",
                             "glasses", "hair", "background", "foreground", "several", "group"}
    def _is_real_name(s):
        words = s.lower().split()
        return len(words) <= 3 and not any(w in description_keywords for w in words)

    real_names    = [n for n in all_gemini if _is_real_name(n)]
    unassigned_names = list(all_gemini)  # all names available for assignment

    for face in faces:
        match = match_face(face["embedding"], known_people, threshold=0.5)
        if match:
            matched_name = match.get("name", "")
            confidence = match["match_score"]
            # Remove from unassigned if Gemini also mentioned this exact name
            if matched_name in unassigned_names:
                unassigned_names.remove(matched_name)
            # Rename if: current name is unknown_N or a description, and a real name is available
            is_placeholder = matched_name.startswith("unknown_") or not _is_real_name(matched_name)
            if is_placeholder and real_names:
                new_name = real_names.pop(0)
                if new_name in unassigned_names:
                    unassigned_names.remove(new_name)
                update_person(match["id"], name=new_name)
                print(f"  Renamed {matched_name} → {new_name} (from Gemini context)")
            else:
                print(f"  Known person: {matched_name} (score: {confidence:.2f})")
            update_person(match["id"], last_seen=timestamp)
            seen_persons.append((match["id"], confidence))
        else:
            # New face — prefer a real name, then description, then unknown_N
            if real_names:
                name = real_names.pop(0)
                if name in unassigned_names:
                    unassigned_names.remove(name)
            elif unassigned_names:
                name = unassigned_names.pop(0)
            else:
                name = f"unknown_{len(known_people) + 1}"
            person_id = save_person({
                "name": name,
                "face_embedding": face["embedding"],
                "thumbnail_path": face.get("crop_path", ""),
                "first_seen": timestamp,
                "last_seen": timestamp,
            })
            print(f"  New person saved as: {name}")
            seen_persons.append((person_id, 1.0))
            known_people = get_all_people()  # refresh

    return seen_persons


def _link_faces_to_memory(seen_persons: list[tuple], memory_id: int | None):
    """Link people seen in this chunk to the memory via the join table.
    seen_persons is a list of (person_id, confidence) tuples."""
    if not memory_id or not seen_persons:
        return
    for person_id, confidence in seen_persons:
        link_person_memory(person_id, memory_id, confidence)
