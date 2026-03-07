import os
import sys
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis

EMBEDDINGS_FILE = "embeddings.json"
MATCH_THRESHOLD = 0.45  # cosine similarity threshold for face match


def init_face_app():
    app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    return app


def get_face_embedding(app, image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)
    if not faces:
        print(f"  No face detected in {image_path}")
        return None
    return faces[0].embedding.tolist()


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def load_store():
    if os.path.exists(EMBEDDINGS_FILE):
        with open(EMBEDDINGS_FILE) as f:
            return json.load(f)
    return {}


def save_store(store):
    with open(EMBEDDINGS_FILE, "w") as f:
        json.dump(store, f, indent=2)


def search(image_path):
    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    store = load_store()
    query_name = os.path.basename(image_path)

    print(f"Initializing face model...")
    app = init_face_app()

    print(f"Generating face embedding for {query_name}...")
    embedding = get_face_embedding(app, image_path)
    if embedding is None:
        return

    print(f"Embedding dimension: {len(embedding)}")

    if store:
        similarities = []
        for name, entry in store.items():
            sim = cosine_similarity(embedding, entry["embedding"])
            similarities.append((name, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)

        print(f"\n--- Results for '{query_name}' ---")
        for name, sim in similarities:
            print(f"  {name}: {sim:.4f}")

        best_name, best_sim = similarities[0]
        if best_sim >= MATCH_THRESHOLD:
            print(f"\nMATCH FOUND: '{best_name}' (similarity: {best_sim:.4f})")
            return best_name, best_sim

    print(f"\nNo match found (threshold: {MATCH_THRESHOLD}). Adding '{query_name}' to store.")
    store[query_name] = {"embedding": embedding}
    save_store(store)
    print(f"Stored. Total images in store: {len(store)}")
    return None, None


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_embeddings.py <image_path>")
        sys.exit(1)

    search(sys.argv[1])
