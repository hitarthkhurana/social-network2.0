# Social Network 2.0 — Person-Indexed Video Memory System

A system that processes videos, detects faces, transcribes audio, and builds a searchable memory store indexed by person. When a person appears again in a new video, the system recognizes them and retrieves all their existing context.

## How It Works

```
Video (.mp4)
    │
    ├── Extract frames → InsightFace detects faces
    │       ├── Known person? → Retrieve existing context
    │       └── New person? → Store face embedding
    │
    ├── Extract audio → Gemini transcribes speech
    │
    └── Frames + transcript → Gemini analyzes & summarizes
            │
            └── Store memory + transcript linked to detected persons
```

## Setup

```bash
pip install -r requirements.txt
brew install ffmpeg
```

Create a `.env` file:
```
GEMINI_API_KEY=your-key-here
```

## Usage

### Ingest a video
```bash
python ingest.py video.mp4
```

The system will:
1. Extract frames and detect all faces
2. Check each face against the store — print existing context if recognized, or add as new
3. Transcribe audio using Gemini
4. Generate a summary and store everything linked to the detected persons

### Search by face
```bash
# Find a person and show all their memories
python query.py face photo.jpg

# Ask a question about a person
python query.py face photo.jpg "what did they talk about?"

# Label a person
python query.py face photo.jpg --name "Alice"
```

### Other commands
```bash
python query.py persons          # List all known persons
python query.py search "topic"   # Text search across all memories
python query.py chat             # Interactive chat
python query.py timeline         # Show all memories chronologically
python query.py stats            # Storage statistics
```

## Architecture

| File | Purpose |
|------|---------|
| `config.py` | Settings, model names, paths, thresholds |
| `storage.py` | SQLite DB: persons, memories, person_memories tables |
| `faces.py` | InsightFace face detection, embedding, matching |
| `processor.py` | Gemini analysis + face detection per video chunk |
| `ingest.py` | Video processing pipeline |
| `query.py` | CLI for searching by face, text, or interactive chat |

## Tech Stack
- **Face Recognition**: InsightFace (buffalo_l, 512-dim embeddings)
- **Audio Transcription**: Gemini 2.0 Flash
- **Video Analysis**: Gemini 2.5 Flash Lite
- **Text Embeddings**: Gemini Embedding 001 (768-dim)
- **Storage**: SQLite
