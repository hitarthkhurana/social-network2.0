"""
faces.py — Face detection, embedding, and identity matching using InsightFace.

Pipeline:
  Video frame → detect faces → 512-dim embedding per face
  On query: compare face embedding vs all stored faces → find match
"""

import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Load once at import time
_app = None

def get_face_app() -> FaceAnalysis:
    global _app
    if _app is None:
        _app = FaceAnalysis(providers=["CPUExecutionProvider"])
        _app.prepare(ctx_id=0, det_size=(640, 640))
    return _app


def detect_faces(image_path: str) -> list[dict]:
    """
    Detect all faces in an image.
    Returns list of:
      {
        embedding: np.ndarray (512-dim),
        bbox: [x1, y1, x2, y2],
        confidence: float,
        crop_path: str  (saved face crop)
      }
    """
    app = get_face_app()
    img = cv2.imread(image_path)
    if img is None:
        return []

    faces = app.get(img)
    results = []

    for i, face in enumerate(faces):
        if face.embedding is None:
            continue

        # Save face crop
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        # Add padding
        h, w = img.shape[:2]
        pad = 20
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)

        crop = img[y1:y2, x1:x2]
        crop_path = image_path.replace(".jpg", f"_face_{i}.jpg")
        cv2.imwrite(crop_path, crop)

        results.append({
            "embedding": np.array(face.embedding, dtype=np.float32),
            "bbox": [x1, y1, x2, y2],
            "confidence": float(face.det_score),
            "crop_path": crop_path
        })

    return results


def detect_faces_from_frame(frame: np.ndarray) -> list[dict]:
    """Detect faces directly from a cv2 frame (no file needed)."""
    app = get_face_app()
    faces = app.get(frame)
    results = []
    for face in faces:
        if face.embedding is not None:
            results.append({
                "embedding": np.array(face.embedding, dtype=np.float32),
                "bbox": [int(v) for v in face.bbox],
                "confidence": float(face.det_score)
            })
    return results


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def match_face(query_embedding: np.ndarray, stored_people: list[dict], threshold: float = 0.5) -> dict | None:
    """
    Find the best matching person for a query face embedding.
    Returns the matched person dict or None if no match above threshold.
    """
    best_score = -1
    best_person = None

    for person in stored_people:
        emb = person.get("face_embedding")
        if emb is None:
            continue
        score = cosine_similarity(query_embedding, emb)
        if score > best_score:
            best_score = score
            best_person = person

    if best_score >= threshold:
        best_person["match_score"] = best_score
        return best_person
    return None
