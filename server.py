"""
server.py — FastAPI server for Meta glasses integration.

Endpoints:
    POST /identify  — JPEG image → face match → name + details
    POST /enroll    — MP4 video  → extract face + transcribe → store

Run:
    python server.py
"""

import os
import tempfile
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from config import GEMINI_API_KEY, FACE_MATCH_THRESHOLD, TEMP_DIR
from storage import (
    init_db, find_matching_person, get_memories_for_person,
    get_all_persons, save_person, save_memory, link_person_memory,
    update_person_label,
)
from faces import init_face_app, detect_faces, crop_face_thumbnail
from config import FACE_THUMBNAILS_DIR

from google import genai
from google.genai import types

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

face_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global face_app
    print("Loading face recognition model...")
    init_db()
    face_app = init_face_app()
    print("Server ready.")
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/identify")
async def identify(request: Request):
    """Receive JPEG bytes, identify the person."""
    image_bytes = await request.body()

    # Decode JPEG to image
    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"name": None, "details": None}, status_code=400)

    # Detect faces
    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"name": None, "details": None})

    # Get embedding of largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = np.array(face.embedding, dtype=np.float32)

    # Search store
    match = find_matching_person(embedding, FACE_MATCH_THRESHOLD)
    if not match:
        return JSONResponse({"name": None, "details": None})

    person_id, similarity = match

    # Get person label
    persons = get_all_persons()
    person = next((p for p in persons if p["id"] == person_id), None)
    name = person["label"] if person and person.get("label") else None

    # Gather context from all memories and synthesize a concise summary
    memories = get_memories_for_person(person_id)
    context_parts = []
    for m in memories:
        if m.get("transcript") and m["transcript"] != "No speech detected.":
            context_parts.append(m["transcript"])
        if m.get("summary"):
            context_parts.append(m["summary"])

    if not context_parts:
        return JSONResponse({"name": name, "details": None})

    raw_context = "\n".join(context_parts)
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"Summarize everything known about this person in 1-2 short sentences "
                f"for a quick verbal reminder through smart glasses. Be concise and natural.\n\n"
                f"Person name: {name or 'Unknown'}\n"
                f"Context:\n{raw_context[:1000]}"
            ),
        )
        details = response.text.strip()
    except Exception:
        details = raw_context[:200]

    return JSONResponse({"name": name, "details": details})


@app.post("/enroll")
async def enroll(video: UploadFile = File(...)):
    """Receive MP4 video, extract face + transcribe audio, store."""
    # Save uploaded video to temp file
    tmp_video = os.path.join(TEMP_DIR, f"enroll_{video.filename}")
    with open(tmp_video, "wb") as f:
        f.write(await video.read())

    try:
        # Step 1: Extract a frame for face detection
        cap = cv2.VideoCapture(tmp_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30

        # Sample a few frames to find the best face
        best_face = None
        best_frame_path = None
        best_area = 0

        sample_points = [int(total_frames * p) for p in [0.2, 0.4, 0.5, 0.6, 0.8]]
        for frame_idx in sample_points:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = face_app.get(frame)
            for face in faces:
                area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                if area > best_area:
                    best_area = area
                    best_face = face
                    frame_path = os.path.join(TEMP_DIR, f"enroll_frame_{frame_idx}.jpg")
                    cv2.imwrite(frame_path, frame)
                    best_frame_path = frame_path

        cap.release()

        if best_face is None:
            return JSONResponse({"status": "error", "name": None, "message": "No face detected in video"}, status_code=400)

        # Step 2: Extract audio and transcribe with Gemini
        audio_path = os.path.join(TEMP_DIR, "enroll_audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", tmp_video, "-ac", "1", "-ar", "8000", audio_path],
            capture_output=True,
        )

        transcript = ""
        if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
            with open(audio_path, "rb") as f:
                audio_data = f.read()

            response = gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=types.Content(
                    parts=[
                        types.Part(inline_data=types.Blob(mime_type="audio/wav", data=audio_data)),
                        types.Part(text=(
                            "This audio is someone describing a person they just met. "
                            "Extract the person's NAME and any DETAILS mentioned about them. "
                            "Return ONLY in this format: Name: <name>\nDetails: <details>\n"
                            "If no name is mentioned, return Name: Unknown"
                        )),
                    ]
                ),
            )
            transcript = response.text.strip()

        # Parse name and details from transcript
        name = None
        details = None
        for line in transcript.split("\n"):
            line = line.strip()
            if line.lower().startswith("name:"):
                name = line[5:].strip()
                if name.lower() == "unknown":
                    name = None
            elif line.lower().startswith("details:"):
                details = line[8:].strip()

        # Step 3: Check if person already exists
        embedding = np.array(best_face.embedding, dtype=np.float32)
        match = find_matching_person(embedding, FACE_MATCH_THRESHOLD)

        if match:
            person_id, _ = match
            if name:
                update_person_label(person_id, name)
        else:
            # Save new person
            thumbnail_path = os.path.join(
                FACE_THUMBNAILS_DIR, f"person_{hash(embedding.tobytes()) & 0xFFFFFFFF:08x}.jpg"
            )
            if best_frame_path:
                crop_face_thumbnail(best_frame_path, best_face.bbox.tolist(), thumbnail_path)
            person_id = save_person(embedding, thumbnail_path, label=name)

        # Step 4: Save memory with transcript
        from processor import get_embedding
        summary = f"{name or 'Unknown person'}: {details}" if details else f"Enrolled {name or 'unknown person'}"
        text_embedding = get_embedding(summary)

        memory = {
            "source": "glasses_enroll",
            "summary": summary,
            "detail_level": "high",
            "people": [name] if name else [],
            "activity": "enrollment",
            "tags": ["enrollment", "glasses"],
            "transcript": transcript,
            "importance": 0.8,
            "keyframe_paths": [],
            "embedding": text_embedding,
        }
        memory_id = save_memory(memory)
        link_person_memory(person_id, memory_id, 1.0)

        # Cleanup
        for f in [tmp_video, audio_path, best_frame_path]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        return JSONResponse({"status": "saved", "name": name or "Unknown"})

    except Exception as e:
        # Cleanup on error
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        return JSONResponse({"status": "error", "name": None, "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import socket
    import subprocess as sp
    try:
        result = sp.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True)
        local_ip = result.stdout.strip() or "localhost"
    except Exception:
        local_ip = "localhost"
    print(f"\n  Server starting on http://{local_ip}:8000")
    print(f"  Tell your teammates: http://{local_ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
