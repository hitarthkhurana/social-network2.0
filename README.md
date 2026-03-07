# Engram — AI Memory Compaction System

A semantic memory compressor for video/audio content. Instead of storing raw video, Engram extracts meaning — compressing hours of footage into a tiny, queryable memory database.

Built for the **YC x Google DeepMind Multimodal Frontier Hackathon**.

---

## The Core Idea

Raw video is huge and unsearchable. Engram processes video through AI and stores only what matters:

```
Raw Video (1GB/hr)  →  Engram  →  Semantic Memories (~34KB/hr)
```

Every moment gets an **importance score**. High-importance moments get detailed summaries + keyframe photos. Low-importance moments get a one-liner. Nothing is ever fully discarded — the timeline stays complete.

---

## How It Works

### Full Pipeline

```
YouTube URL
    ↓
yt-dlp → .mp4 downloaded to ~/.memory-compactor/temp/
    ↓
ffmpeg → .wav audio extracted
    ↓
Whisper (on-device) → transcribes in 30s windows → timestamped segments
    ↓
Split into 60s chunks
    ↓
For each chunk:
    ├── Extract frames every 10s
    ├── Get transcript for this 60s window
    │
    └── API Call 1 → Gemini (score + summarize in one shot)
          sends: 2 frames + transcript
          returns: {
            importance: 0.0–1.0,
            summary: "...",
            people: [...],
            activity: "...",
            tags: [...]
          }
          │
          ├── importance ≥ 0.5 → HIGH: full summary, store transcript, 2–5 keyframes
          └── importance < 0.5 → LOW:  one-liner, no transcript, 0–1 keyframes
    │
    └── API Call 2 → Gemini Embeddings
          sends: summary text
          returns: 768-dimensional vector
    │
    └── Save to SQLite
          └── Delete all temp files
```

### Query Pipeline

```
User query (text)
    ↓
API Call → Gemini Embeddings → 768-dimensional vector
    ↓
Cosine similarity vs all stored memory embeddings (local, no API)
    ↓
Top 5 closest memories retrieved from SQLite
    ↓
API Call → Gemini Flash → synthesized natural language answer
    ↓
Answer + memory cards printed to terminal
```

---

## What Gets Stored Per Memory

| Field | High Importance (≥0.5) | Low Importance (<0.5) |
|-------|----------------------|----------------------|
| Summary | 2–3 sentences | 1 sentence |
| People | yes | no |
| Activity | yes | no |
| Tags | 3–5 | 1–2 |
| Transcript | full text | none |
| Keyframes | 2–5 JPEG thumbnails (320×240) | 0–1 |
| Embedding | 768 floats | 768 floats |

### Keyframes by importance score

| Score | Keyframes saved |
|-------|----------------|
| 0.8 – 1.0 | 5 |
| 0.5 – 0.8 | 2 |
| 0.2 – 0.5 | 1 |
| 0.0 – 0.2 | 0 |

---

## File Structure

```
memory-compactor/
├── config.py        # All settings — models, thresholds, paths
├── ingest.py        # Download + transcribe + chunk + orchestrate
├── processor.py     # Gemini calls — analyze (score+summarize) + embed
├── storage.py       # SQLite read/write + cosine similarity search
├── reconstruct.py   # NanoBanana 2 image reconstruction (next iteration)
├── query.py         # CLI interface — chat, search, timeline, stats
└── requirements.txt

~/.memory-compactor/
├── memories.db      # SQLite database
├── keyframes/       # Thumbnail JPEGs
├── reconstructed/   # NanoBanana generated images
└── temp/            # Temporary files (auto-deleted)
```

---

## Setup

### Requirements
- Python 3.9+
- ffmpeg
- Mac (for now)

### Install

```bash
brew install ffmpeg

pip install google-genai yt-dlp opencv-python openai-whisper rich numpy Pillow
```

### API Key

Get a Gemini API key from [aistudio.google.com](https://aistudio.google.com).

Set it as an environment variable (recommended):
```bash
export GEMINI_API_KEY="your-key-here"
```

Or edit `config.py` directly (not recommended for shared repos).

> **Note:** Free tier keys have low quotas (~10–15 requests/day). Add billing for real usage — costs fractions of a cent per video.

---

## Usage

### Ingest a YouTube video
```bash
python ingest.py "https://www.youtube.com/watch?v=VIDEO_ID"

# Multiple videos
python ingest.py "url1" "url2" "url3"

# From a file (one URL per line)
python ingest.py --file urls.txt
```

### Query your memories
```bash
# Interactive chat (recommended)
python query.py chat

# One-shot search
python query.py search "what habits were discussed?"

# Chronological timeline of all memories
python query.py timeline

# Storage stats
python query.py stats
```

---

## Key Design Decisions

### Why not store raw video?
Storage is expensive. More importantly, raw video is unsearchable. By converting to semantic summaries + embeddings, we get both compression and instant natural language search.

### Why Whisper in 30-second windows?
Whisper's context window is 30 seconds. Feeding longer audio causes hallucination and text repetition. We split audio into 30s chunks, transcribe each independently, then stitch timestamps back.

### Why combine importance scoring + summarization into one API call?
Originally 3 API calls per chunk (score → summarize → embed). Scoring and summarizing were sending the same frames + transcript twice. Combined into 1 call that does both, saving 33% of API calls.

### Why cosine similarity instead of a vector database?
For small datasets (< 10K memories), brute-force cosine similarity over numpy arrays is fast enough and has zero dependencies. When scaling up, swap `storage.py`'s search for Qdrant or pgvector.

### Why JPEG thumbnails at 320×240?
Balance between visual context and storage size. A full 1080p frame is ~500KB. At 320×240 quality 70, it's ~15KB — 33x smaller while still recognizable.

---

## API Calls Budget

| Operation | Calls | What's sent |
|-----------|-------|-------------|
| Per chunk | 1 | 2 frames + transcript |
| Per chunk embedding | 1 | summary text only |
| Per query | 1 | query text only |
| Per answer synthesis | 1 | 5 summaries + query |

**3-min video = ~8 API calls total**

---

## Next Iteration (In This Hackathon)

- [ ] Stitch stored keyframes into short video clips using NanoBanana 2
- [ ] Live screen + webcam + microphone capture (not just YouTube)
- [ ] Ray-Ban glasses integration as input source

## Future (Post-Hackathon)

- [ ] Swap SQLite embeddings for Qdrant vector DB at scale
- [ ] Mobile app for capture
- [ ] Multi-user support
