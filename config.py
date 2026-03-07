import os

# Vertex AI — everything runs through GCP project, no AI Studio key needed
VERTEX_PROJECT  = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-gcp-project-id")
VERTEX_LOCATION = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")

GEMINI_FLASH_MODEL = "gemini-3.1-flash-lite"   # via Vertex AI
EMBEDDING_MODEL    = "multimodalembedding@001"  # 1408-dim, image + text

# Importance thresholds
IMPORTANCE_HIGH = 0.5
IMPORTANCE_LOW  = 0.2

# Keyframes per importance band (used for thumbnail extraction)
KEYFRAMES_BY_IMPORTANCE = {
    (0.8, 1.0): 5,
    (0.5, 0.8): 2,
    (0.2, 0.5): 1,
    (0.0, 0.2): 0,
}

# How many seconds of video to process as one memory chunk
CHUNK_DURATION_SEC = 60

DB_PATH           = os.path.expanduser("~/.memory-compactor/memories.db")
CLIPS_DIR         = os.path.expanduser("~/.memory-compactor/clips")
RECONSTRUCTED_DIR = os.path.expanduser("~/.memory-compactor/reconstructed")
TEMP_DIR          = os.path.expanduser("~/.memory-compactor/temp")
