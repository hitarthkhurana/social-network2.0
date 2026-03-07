import os

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY environment variable. Get a key at https://aistudio.google.com")

GEMINI_FLASH_MODEL   = "gemini-2.5-flash-lite"   # best free tier limits
GEMINI_MODEL         = "gemini-2.5-flash-lite"   # use flash-lite to save tokens
NANOBANANA_MODEL     = "gemini-2.0-flash-exp-image-generation"
EMBEDDING_MODEL      = "models/gemini-embedding-001"

# Importance thresholds
IMPORTANCE_HIGH = 0.5
IMPORTANCE_LOW  = 0.2

# Keyframes per importance band
KEYFRAMES_BY_IMPORTANCE = {
    (0.8, 1.0): 5,
    (0.5, 0.8): 2,
    (0.2, 0.5): 1,
    (0.0, 0.2): 0,
}

# Frame extraction: 1 frame every N seconds from video
FRAME_INTERVAL_SEC = 10

# How many seconds of video to process as one memory chunk
CHUNK_DURATION_SEC = 60

# Max frames sent to Gemini per API call (keeps token usage low)
MAX_FRAMES_TO_GEMINI = 2

DB_PATH           = os.path.expanduser("~/.memory-compactor/memories.db")
KEYFRAMES_DIR     = os.path.expanduser("~/.memory-compactor/keyframes")
RECONSTRUCTED_DIR = os.path.expanduser("~/.memory-compactor/reconstructed")
TEMP_DIR          = os.path.expanduser("~/.memory-compactor/temp")
