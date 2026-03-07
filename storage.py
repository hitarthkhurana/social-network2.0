import sqlite3
import json
import os
import numpy as np
from datetime import datetime
from config import DB_PATH, CLIPS_DIR, RECONSTRUCTED_DIR, TEMP_DIR


def init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    os.makedirs(CLIPS_DIR, exist_ok=True)
    os.makedirs(RECONSTRUCTED_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp    TEXT NOT NULL,
            source       TEXT,
            summary      TEXT NOT NULL,
            detail_level TEXT NOT NULL,
            people       TEXT,
            activity     TEXT,
            tags         TEXT,
            importance   REAL NOT NULL,
            clip_paths   TEXT,
            segments     TEXT,
            embedding    BLOB
        )
    """)
    conn.commit()
    conn.close()


def save_memory(memory: dict):
    conn = sqlite3.connect(DB_PATH)
    embedding = memory.get("embedding")
    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

    conn.execute("""
        INSERT INTO memories
        (timestamp, source, summary, detail_level, people, activity, tags,
         importance, clip_paths, segments, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        memory.get("timestamp", datetime.utcnow().isoformat()),
        memory.get("source", ""),
        memory["summary"],
        memory.get("detail_level", "low"),
        json.dumps(memory.get("people", [])),
        memory.get("activity", ""),
        json.dumps(memory.get("tags", [])),
        memory["importance"],
        json.dumps(memory.get("clip_paths", [])),
        json.dumps(memory.get("segments", [])),
        embedding_blob,
    ))
    conn.commit()
    conn.close()


def get_all_memories() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM memories ORDER BY timestamp ASC").fetchall()
    conn.close()
    return [_row_to_dict(r) for r in rows]


def search_by_embedding(query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
    memories = get_all_memories()
    scored = []
    for m in memories:
        if m.get("embedding") is not None:
            sim = cosine_similarity(query_embedding, m["embedding"])
            scored.append((sim, m))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in scored[:top_k]]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.array(a, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def get_stats() -> dict:
    conn = sqlite3.connect(DB_PATH)
    total   = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    high    = conn.execute("SELECT COUNT(*) FROM memories WHERE importance >= 0.5").fetchone()[0]
    low     = conn.execute("SELECT COUNT(*) FROM memories WHERE importance < 0.5").fetchone()[0]
    avg_imp = conn.execute("SELECT AVG(importance) FROM memories").fetchone()[0] or 0
    conn.close()

    db_size = os.path.getsize(DB_PATH) if os.path.exists(DB_PATH) else 0
    clips_size = sum(
        os.path.getsize(os.path.join(CLIPS_DIR, f))
        for f in os.listdir(CLIPS_DIR)
        if os.path.isfile(os.path.join(CLIPS_DIR, f))
    ) if os.path.exists(CLIPS_DIR) else 0

    return {
        "total_memories": total,
        "high_importance": high,
        "low_importance": low,
        "avg_importance": round(avg_imp, 2),
        "db_size_kb": round(db_size / 1024, 1),
        "clips_size_kb": round(clips_size / 1024, 1),
    }


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["people"]     = json.loads(d["people"] or "[]")
    d["tags"]       = json.loads(d["tags"] or "[]")
    d["clip_paths"] = json.loads(d["clip_paths"] or "[]")
    d["segments"]   = json.loads(d["segments"] or "[]")
    if d["embedding"]:
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
    return d
