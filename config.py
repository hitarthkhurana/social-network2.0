import os
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env file")

GEMINI_FLASH_MODEL = "gemini-2.5-flash-lite"
EMBEDDING_MODEL = "models/gemini-embedding-001"

FACE_MATCH_THRESHOLD = 0.45

IMPORTANCE_HIGH = 0.5
IMPORTANCE_LOW = 0.2

KEYFRAMES_BY_IMPORTANCE = {
    (0.8, 1.0): 5,
    (0.5, 0.8): 2,
    (0.2, 0.5): 1,
    (0.0, 0.2): 0,
}

FRAME_INTERVAL_SEC = 10
CHUNK_DURATION_SEC = 60
MAX_FRAMES_TO_GEMINI = 2

DB_PATH = os.path.expanduser("~/.memory-compactor/memories.db")
KEYFRAMES_DIR = os.path.expanduser("~/.memory-compactor/keyframes")
FACE_THUMBNAILS_DIR = os.path.expanduser("~/.memory-compactor/face_thumbnails")
TEMP_DIR = os.path.expanduser("~/.memory-compactor/temp")
