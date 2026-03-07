# Engram — AI Memory Compaction System

Semantic memory compressor for video. Instead of storing raw video, Engram processes it through AI and stores structured memories — tiny, queryable, and searchable by face or text.

Built for the **YC x Google DeepMind Multimodal Frontier Hackathon**.

---

## Setup

### 1. System dependencies
```bash
brew install ffmpeg
```

### 2. Python packages
```bash
pip install -r requirements.txt
```

> InsightFace will auto-download the `buffalo_l` face model (~200MB) on first run.

### 3. Google Cloud (Vertex AI)
Everything runs through Vertex AI — no API keys needed, uses your GCP project credentials.

```bash
# One-time auth setup
gcloud auth login
gcloud auth application-default login
gcloud auth application-default set-quota-project <your-project-id>
gcloud config set project <your-project-id>
```

That's it. The code auto-detects your active gcloud project.

---

## Usage

### Ingest a YouTube video
```bash
python ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"
```

### Ingest a local video file
```bash
python ingest.py /path/to/video.mp4
```

### Record from webcam and ingest
```bash
python record.py              # records until you press Q
python record.py --seconds 30 # records for 30 seconds
python record.py --no-ingest  # record only, don't process
```

### Query your memories
```bash
python query.py chat                        # interactive chat (recommended)
python query.py search "what was discussed" # one-shot semantic search
python query.py timeline                    # all memories chronologically
python query.py stats                       # storage stats
python query.py people                      # list all known people
python query.py person photo.jpg            # look up a person by photo
python query.py person photo.jpg --name "Alice"  # look up + name them
```

---

## Pipeline

```
Video (.mp4 / YouTube URL / webcam)
    ↓
ffmpeg → split into 60s chunks
    ↓
For each chunk → Gemini 2.5 Flash Lite (raw video bytes, audio included)
    Returns: importance score, summary, scene description,
             verbatim transcript, people, activity, tags,
             exact timestamps of important moments
    ↓
ffmpeg → cut clips at important timestamps
    ↓
InsightFace → detect faces in keyframe
    ├── Known face? → match by cosine similarity, update last_seen
    └── New face?   → store embedding, auto-name from Gemini people list
    ↓
Vertex AI multimodalembedding@001 → 1408-dim embedding (image + text)
    ↓
SQLite → save memory + link persons via person_memories join table
    ↓
Delete original video and temp files
```

### Query pipeline
```
Your question
    ↓
Vertex AI embedding → 1408-dim vector
    ↓
Cosine similarity vs all stored memory embeddings (local numpy, no API)
    ↓
Top 5 memories → Gemini synthesizes a natural language answer
```

---

## File Structure

```
memory-compactor/
├── config.py        # Models, thresholds, paths — auto-detects GCP project
├── ingest.py        # YouTube + local video ingestion pipeline
├── record.py        # Webcam recording + ingest
├── processor.py     # Gemini analysis + face detection + embeddings
├── faces.py         # InsightFace detection + batched matching
├── storage.py       # SQLite: memories, persons, person_memories tables
├── query.py         # CLI: chat, search, timeline, person lookup
├── reconstruct.py   # Image reconstruction (future)
└── requirements.txt

~/.memory-compactor/
├── memories.db      # SQLite database
├── clips/           # Video clips of important moments
├── reconstructed/   # AI-generated reconstructions
└── temp/            # Auto-deleted working files
```

---

## Database Schema

```sql
memories        — one row per 60s chunk (summary, scene, transcript, embedding, importance)
persons         — one row per unique face (name, embedding, first/last seen)
person_memories — join table (person_id, memory_id, confidence)
```

---

## Tech Stack

| Component | Tech |
|---|---|
| Video analysis + transcription | Gemini 2.5 Flash Lite via Vertex AI |
| Multimodal embeddings | `multimodalembedding@001` — 1408-dim image+text |
| Face recognition | InsightFace `buffalo_l` — 512-dim embeddings |
| Storage | SQLite |
| Auth | Google Cloud ADC (no API keys) |
