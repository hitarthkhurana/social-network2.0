"""
server.py — FastAPI server for Meta glasses integration.

Endpoints:
    POST /identify  — JPEG image → face match → name + details
    POST /enroll    — MP4 video  → extract face + transcribe → store

Run:
    python server.py
"""

import os
import subprocess
from contextlib import asynccontextmanager

import cv2
import numpy as np
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import JSONResponse
import uvicorn

from config import VERTEX_PROJECT, VERTEX_LOCATION, GEMINI_FLASH_MODEL, FACE_THUMBNAILS_DIR, TEMP_DIR
from storage import (
    init_db, find_matching_person, get_memories_for_person,
    get_all_persons, save_person, save_memory, link_person_memory,
    update_person_label,
)
from faces import init_face_app, detect_faces, crop_face_thumbnail

vertexai.init(project=VERTEX_PROJECT, location=VERTEX_LOCATION)
gemini = GenerativeModel(GEMINI_FLASH_MODEL)

FACE_MATCH_THRESHOLD = 0.45
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
    """Receive JPEG bytes → identify person → return name + context summary."""
    image_bytes = await request.body()

    img_array = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"name": None, "details": None}, status_code=400)

    faces = face_app.get(img)
    if not faces:
        return JSONResponse({"name": None, "details": None})

    # Largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    embedding = np.array(face.embedding, dtype=np.float32)

    match = find_matching_person(embedding, FACE_MATCH_THRESHOLD)
    if not match:
        return JSONResponse({"name": None, "details": None})

    person_id, similarity = match
    persons = get_all_persons()
    person = next((p for p in persons if p["id"] == person_id), None)
    name = person.get("name") if person else None

    memories = get_memories_for_person(person_id)
    context_parts = []
    for m in memories:
        if m.get("transcript"):
            context_parts.append(m["transcript"])
        if m.get("summary"):
            context_parts.append(m["summary"])

    if not context_parts:
        return JSONResponse({"name": name, "details": None})

    raw_context = "\n".join(context_parts)
    try:
        response = gemini.generate_content(
            f"Summarize everything known about this person in 1-2 short sentences "
            f"for a quick verbal reminder through smart glasses. Be concise and natural.\n\n"
            f"Person name: {name or 'Unknown'}\n"
            f"Context:\n{raw_context[:1000]}"
        )
        details = response.text.strip()
    except Exception:
        details = raw_context[:200]

    return JSONResponse({"name": name, "details": details})


@app.post("/enroll")
async def enroll(video: UploadFile = File(...)):
    """Receive MP4 video → extract face + transcribe audio → store person."""
    tmp_video = os.path.join(TEMP_DIR, f"enroll_{video.filename}")
    with open(tmp_video, "wb") as f:
        f.write(await video.read())

    try:
        # Sample frames to find best face
        cap = cv2.VideoCapture(tmp_video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        best_face = None
        best_frame_path = None
        best_area = 0

        for p in [0.2, 0.4, 0.5, 0.6, 0.8]:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(total_frames * p))
            ret, frame = cap.read()
            if not ret:
                continue
            for face in face_app.get(frame):
                area = (face.bbox[2] - face.bbox[0]) * (face.bbox[3] - face.bbox[1])
                if area > best_area:
                    best_area = area
                    best_face = face
                    frame_path = os.path.join(TEMP_DIR, f"enroll_frame_{int(p*100)}.jpg")
                    cv2.imwrite(frame_path, frame)
                    best_frame_path = frame_path
        cap.release()

        if best_face is None:
            return JSONResponse({"status": "error", "message": "No face detected"}, status_code=400)

        # Transcribe audio with Gemini (send raw video bytes — no separate extraction)
        with open(tmp_video, "rb") as f:
            video_bytes = f.read()

        transcript = ""
        try:
            video_part = Part.from_data(data=video_bytes, mime_type="video/mp4")
            response = gemini.generate_content([
                video_part,
                "This video is someone describing a person they just met. "
                "Extract the person's NAME and any DETAILS mentioned about them. "
                "Return ONLY:\nName: <name>\nDetails: <details>\n"
                "If no name mentioned, return Name: Unknown"
            ])
            transcript = response.text.strip()
        except Exception:
            pass

        # Parse name and details
        name = None
        details = None
        for line in transcript.split("\n"):
            if line.lower().startswith("name:"):
                name = line[5:].strip()
                if name.lower() == "unknown":
                    name = None
            elif line.lower().startswith("details:"):
                details = line[8:].strip()

        # Match or create person
        embedding = np.array(best_face.embedding, dtype=np.float32)
        match = find_matching_person(embedding, FACE_MATCH_THRESHOLD)

        if match:
            person_id, _ = match
            if name:
                update_person_label(person_id, name)
        else:
            thumbnail_path = os.path.join(
                FACE_THUMBNAILS_DIR, f"person_{hash(embedding.tobytes()) & 0xFFFFFFFF:08x}.jpg"
            )
            if best_frame_path:
                crop_face_thumbnail(best_frame_path, best_face.bbox.tolist(), thumbnail_path)
            person_id = save_person({
                "name": name or "unknown",
                "face_embedding": embedding,
                "thumbnail_path": thumbnail_path,
            })

        # Save memory
        from processor import get_embedding
        summary = f"{name or 'Unknown person'}: {details}" if details else f"Enrolled {name or 'unknown person'}"
        memory_id = save_memory({
            "source": "glasses_enroll",
            "summary": summary,
            "transcript": transcript,
            "detail_level": "high",
            "people": [name] if name else [],
            "activity": "enrollment",
            "tags": ["enrollment", "glasses"],
            "importance": 0.8,
            "embedding": get_embedding(summary),
        })
        link_person_memory(person_id, memory_id, 1.0)

        for f in [tmp_video, best_frame_path]:
            if f and os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass

        return JSONResponse({"status": "saved", "name": name or "Unknown"})

    except Exception as e:
        if os.path.exists(tmp_video):
            os.remove(tmp_video)
        return JSONResponse({"status": "error", "message": str(e)}, status_code=500)


if __name__ == "__main__":
    import subprocess as sp
    try:
        result = sp.run(["ipconfig", "getifaddr", "en0"], capture_output=True, text=True)
        local_ip = result.stdout.strip() or "localhost"
    except Exception:
        local_ip = "localhost"
    print(f"\n  Server: http://{local_ip}:8000\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
