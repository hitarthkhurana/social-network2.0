import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

from config import FACE_MATCH_THRESHOLD, FACE_THUMBNAILS_DIR
from storage import (
    find_matching_person, save_person, get_memories_for_person,
    get_all_persons,
)


def init_face_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def detect_faces(app, image_path: str) -> list:
    img = cv2.imread(image_path)
    if img is None:
        return []
    faces = app.get(img)
    return [
        {"embedding": np.array(f.embedding, dtype=np.float32), "bbox": f.bbox.tolist()}
        for f in faces
    ]


def crop_face_thumbnail(image_path: str, bbox, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        return
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = img.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    face_crop = img[y1:y2, x1:x2]
    if face_crop.size > 0:
        face_crop = cv2.resize(face_crop, (96, 96))
        cv2.imwrite(output_path, face_crop)


def get_person_label(person_id: int) -> str:
    persons = get_all_persons()
    for p in persons:
        if p["id"] == person_id:
            return p.get("label") or f"Person #{person_id}"
    return f"Person #{person_id}"


def match_or_create_person(image_path: str, face_data: dict) -> tuple:
    """Returns (person_id, confidence, is_new)."""
    embedding = face_data["embedding"]
    match = find_matching_person(embedding, FACE_MATCH_THRESHOLD)
    if match:
        person_id, confidence = match
        return person_id, confidence, False

    thumbnail_path = os.path.join(
        FACE_THUMBNAILS_DIR, f"person_{hash(embedding.tobytes()) & 0xFFFFFFFF:08x}.jpg"
    )
    crop_face_thumbnail(image_path, face_data["bbox"], thumbnail_path)
    person_id = save_person(embedding, thumbnail_path)
    return person_id, 1.0, True


def detect_and_resolve_persons(app, frame_paths: list) -> list:
    """Detect faces and resolve to persons. Only links the largest (primary) face per frame.
    Still registers all faces in the store, but returns only the primary for memory linking.
    Returns list of (person_id, confidence, is_new)."""
    primary_persons = {}
    all_detected = []

    for frame_path in frame_paths:
        faces = detect_faces(app, frame_path)
        if not faces:
            continue

        # Find the largest face (most likely the speaker / main subject)
        largest = max(faces, key=lambda f: (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))

        # Register all faces in the store (so we recognize them later)
        for face_data in faces:
            person_id, confidence, is_new = match_or_create_person(frame_path, face_data)
            all_detected.append((person_id, confidence, is_new))

        # But only link the primary face to the memory
        primary_id, primary_conf, primary_new = match_or_create_person(frame_path, largest)
        if primary_id not in primary_persons or primary_conf > primary_persons[primary_id][0]:
            primary_persons[primary_id] = (primary_conf, primary_new)

    # Print all detected for logging but return only primary
    return [(pid, conf, is_new) for pid, (conf, is_new) in primary_persons.items()], all_detected
