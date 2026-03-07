import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from config import DB_PATH, KEYFRAMES_DIR, FACE_THUMBNAILS_DIR, TEMP_DIR


def init_db():
    for d in [os.path.dirname(DB_PATH), KEYFRAMES_DIR, FACE_THUMBNAILS_DIR, TEMP_DIR]:
        os.makedirs(d, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            label           TEXT,
            face_embedding  BLOB NOT NULL,
            thumbnail_path  TEXT,
            created_at      TEXT NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            source          TEXT,
            summary         TEXT NOT NULL,
            detail_level    TEXT NOT NULL,
            people          TEXT,
            activity        TEXT,
            tags            TEXT,
            transcript      TEXT,
            importance      REAL NOT NULL,
            keyframe_paths  TEXT,
            embedding       BLOB
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_memories (
            person_id   INTEGER NOT NULL REFERENCES persons(id),
            memory_id   INTEGER NOT NULL REFERENCES memories(id),
            confidence  REAL,
            PRIMARY KEY (person_id, memory_id)
        )
    """)
    conn.commit()
    conn.close()


def save_memory(memory: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    embedding = memory.get("embedding")
    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

    cursor = conn.execute("""
        INSERT INTO memories
        (timestamp, source, summary, detail_level, people, activity, tags, transcript, importance, keyframe_paths, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        memory.get("timestamp", datetime.utcnow().isoformat()),
        memory.get("source", ""),
        memory["summary"],
        memory.get("detail_level", "low"),
        json.dumps(memory.get("people", [])),
        memory.get("activity", ""),
        json.dumps(memory.get("tags", [])),
        memory.get("transcript"),
        memory["importance"],
        json.dumps(memory.get("keyframe_paths", [])),
        embedding_blob,
    ))
    memory_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return memory_id


def save_person(face_embedding: np.ndarray, thumbnail_path: str = None, label: str = None) -> int:
    conn = sqlite3.connect(DB_PATH)
    embedding_blob = face_embedding.astype(np.float32).tobytes()
    cursor = conn.execute(
        "INSERT INTO persons (label, face_embedding, thumbnail_path, created_at) VALUES (?, ?, ?, ?)",
        (label, embedding_blob, thumbnail_path, datetime.utcnow().isoformat()),
    )
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id


def link_person_memory(person_id: int, memory_id: int, confidence: float):
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO person_memories (person_id, memory_id, confidence) VALUES (?, ?, ?)",
        (person_id, memory_id, confidence),
    )
    conn.commit()
    conn.close()


def find_matching_person(face_embedding: np.ndarray, threshold: float):
    persons = get_all_persons()
    if not persons:
        return None

    best_id, best_sim = None, -1.0
    for p in persons:
        sim = cosine_similarity(face_embedding, p["face_embedding"])
        if sim > best_sim:
            best_sim = sim
            best_id = p["id"]

    if best_sim >= threshold:
        return best_id, best_sim
    return None


def get_all_persons() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM persons").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["face_embedding"] = np.frombuffer(d["face_embedding"], dtype=np.float32)
        result.append(d)
    return result


def get_memories_for_person(person_id: int) -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT m.*, pm.confidence
        FROM memories m
        JOIN person_memories pm ON m.id = pm.memory_id
        WHERE pm.person_id = ?
        ORDER BY m.timestamp ASC
    """, (person_id,)).fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def search_by_embedding(query_embedding: np.ndarray, top_k: int = 5) -> list:
    memories = get_all_memories()
    scored = []
    for m in memories:
        if m.get("embedding") is not None:
            sim = cosine_similarity(query_embedding, m["embedding"])
            scored.append((sim, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def get_all_memories() -> list:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM memories ORDER BY timestamp ASC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def update_person_label(person_id: int, label: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("UPDATE persons SET label = ? WHERE id = ?", (label, person_id))
    conn.commit()
    conn.close()


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    high = conn.execute("SELECT COUNT(*) FROM memories WHERE importance >= 0.5").fetchone()[0]
    low = conn.execute("SELECT COUNT(*) FROM memories WHERE importance < 0.5").fetchone()[0]
    avg_imp = conn.execute("SELECT AVG(importance) FROM memories").fetchone()[0] or 0
    person_count = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
    conn.close()

    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    keyframe_size = sum(
        os.path.getsize(os.path.join(KEYFRAMES_DIR, f))
        for f in os.listdir(KEYFRAMES_DIR)
        if os.path.isfile(os.path.join(KEYFRAMES_DIR, f))
    ) if os.path.exists(KEYFRAMES_DIR) else 0

    return {
        "total_memories": total,
        "high_importance": high,
        "low_importance": low,
        "avg_importance": round(avg_imp, 2),
        "persons": person_count,
        "db_size_kb": round(db_size / 1024, 1),
        "keyframes_size_kb": round(keyframe_size / 1024, 1),
    }


def cosine_similarity(a, b) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["people"] = json.loads(d.get("people") or "[]")
    d["tags"] = json.loads(d.get("tags") or "[]")
    d["keyframe_paths"] = json.loads(d.get("keyframe_paths") or "[]")
    if d.get("embedding"):
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
    return d
