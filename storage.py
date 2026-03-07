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
            scene        TEXT,
            transcript   TEXT,
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
    conn.execute("""
        CREATE TABLE IF NOT EXISTS persons (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            name           TEXT,
            face_embedding BLOB NOT NULL,
            thumbnail_path TEXT,
            first_seen     TEXT,
            last_seen      TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS person_memories (
            person_id  INTEGER NOT NULL REFERENCES persons(id),
            memory_id  INTEGER NOT NULL REFERENCES memories(id),
            confidence REAL,
            PRIMARY KEY (person_id, memory_id)
        )
    """)
    conn.commit()

    # Migrations — add columns that may be missing from older DB versions
    existing = {row[1] for row in conn.execute("PRAGMA table_info(memories)").fetchall()}
    for col, definition in [
        ("scene",      "TEXT"),
        ("transcript", "TEXT"),
        ("clip_paths", "TEXT"),
        ("segments",   "TEXT"),
        ("embedding",  "BLOB"),
    ]:
        if col not in existing:
            conn.execute(f"ALTER TABLE memories ADD COLUMN {col} {definition}")
    conn.commit()
    conn.close()


def save_memory(memory: dict) -> int:
    conn = sqlite3.connect(DB_PATH)
    embedding = memory.get("embedding")
    embedding_blob = embedding.astype(np.float32).tobytes() if embedding is not None else None

    cursor = conn.execute("""
        INSERT INTO memories
        (timestamp, source, summary, scene, transcript, detail_level, people, activity, tags,
         importance, clip_paths, segments, embedding)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        memory.get("timestamp", datetime.utcnow().isoformat()),
        memory.get("source", ""),
        memory["summary"],
        memory.get("scene", ""),
        memory.get("transcript", ""),
        memory.get("detail_level", "low"),
        json.dumps(memory.get("people", [])),
        memory.get("activity", ""),
        json.dumps(memory.get("tags", [])),
        memory["importance"],
        json.dumps(memory.get("clip_paths", [])),
        json.dumps(memory.get("segments", [])),
        embedding_blob,
    ))
    memory_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return memory_id


def save_person(person: dict) -> int:
    """Save a new person. Returns person ID."""
    conn = sqlite3.connect(DB_PATH)
    embedding_blob = person["face_embedding"].astype(np.float32).tobytes()
    cursor = conn.execute("""
        INSERT INTO persons (name, face_embedding, thumbnail_path, first_seen, last_seen)
        VALUES (?, ?, ?, ?, ?)
    """, (
        person.get("name", "unknown"),
        embedding_blob,
        person.get("thumbnail_path", person.get("face_crop_path", "")),
        person.get("first_seen", datetime.utcnow().isoformat()),
        person.get("last_seen", datetime.utcnow().isoformat()),
    ))
    person_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return person_id


def link_person_memory(person_id: int, memory_id: int, confidence: float = 1.0):
    """Link a person to a memory with match confidence. Proper join table."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT OR IGNORE INTO person_memories (person_id, memory_id, confidence) VALUES (?, ?, ?)",
        (person_id, memory_id, confidence),
    )
    conn.commit()
    conn.close()


def update_person(person_id: int, name: str = None, last_seen: str = None):
    """Update person name or last_seen timestamp."""
    conn = sqlite3.connect(DB_PATH)
    if name:
        conn.execute("UPDATE persons SET name=? WHERE id=?", (name, person_id))
    if last_seen:
        conn.execute("UPDATE persons SET last_seen=? WHERE id=?", (last_seen, person_id))
    conn.commit()
    conn.close()


def get_all_people() -> list[dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM persons ORDER BY last_seen DESC").fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        d["face_embedding"] = np.frombuffer(d["face_embedding"], dtype=np.float32)
        # Attach memory_ids for backwards compat
        d["memory_ids"] = _get_memory_ids_for_person(d["id"])
        result.append(d)
    return result


def _get_memory_ids_for_person(person_id: int) -> list[int]:
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute(
        "SELECT memory_id FROM person_memories WHERE person_id=? ORDER BY memory_id ASC",
        (person_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_memories_for_person(person_id: int) -> list[dict]:
    """Get all memories linked to a person via the join table."""
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
    persons = conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
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
        "persons": persons,
        "db_size_kb": round(db_size / 1024, 1),
        "clips_size_kb": round(clips_size / 1024, 1),
    }


def find_matching_person(face_embedding: np.ndarray, threshold: float):
    """Find best matching person. Returns (person_id, confidence) or None."""
    from faces import match_face
    people = get_all_people()
    match = match_face(face_embedding, people, threshold=threshold)
    if match:
        return match["id"], match["match_score"]
    return None


def get_all_persons() -> list[dict]:
    """Alias for get_all_people() — compatibility with server.py."""
    return get_all_people()


def update_person_label(person_id: int, label: str):
    """Alias for update_person(name=...) — compatibility with server.py."""
    update_person(person_id, name=label)


def _row_to_dict(row) -> dict:
    d = dict(row)
    d["people"]     = json.loads(d.get("people") or "[]")
    d["tags"]       = json.loads(d.get("tags") or "[]")
    d["clip_paths"] = json.loads(d.get("clip_paths") or "[]")
    d["segments"]   = json.loads(d.get("segments") or "[]")
    if d.get("embedding"):
        d["embedding"] = np.frombuffer(d["embedding"], dtype=np.float32)
    return d
