import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable. Get a key at https://aistudio.google.com")

GEMINI_FLASH_MODEL = "gemini-2.5-flash-lite"

# Vertex AI multimodal embeddings
VERTEX_PROJECT  = "seraphic-jet-489522-n9"
VERTEX_LOCATION = "us-central1"
EMBEDDING_MODEL = "multimodalembedding@001"  # 1408-dim, image + text

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
