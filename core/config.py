"""
config.py
Central configuration for the system.
All env vars, paths, and constants defined.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from  root
load_dotenv()

#  Base Paths 
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
REPO_DIR = DATA_DIR / "repos"
VECTORSTORE_DIR = DATA_DIR / "vectorstore"
EMBEDDING_MODEL_PATH = BASE_DIR / "embedding_model"

#  Target Repository 
REPO_URL = "https://github.com/psf/requests.git"
REPO_NAME = "requests"
REPO_LOCAL_PATH = REPO_DIR / REPO_NAME

#  Groq LLM Config 
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MAX_TOKENS = 4096
GROQ_TEMPERATURE = 0.1          # low temp for factual code Q&A
GROQ_MAX_RETRIES = 3
GROQ_RETRY_WAIT_SECONDS = 30    # wait time when rate limit reached

#  Embedding Config 
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # sentence-transformers
EMBEDDING_DEVICE = "cpu"                # "cuda" for GPU use

#  ChromaDB Config 
CHROMA_COLLECTION_NAME = "codebase_qa"
CHROMA_PERSIST_DIR = str(VECTORSTORE_DIR)

#  Chunking Config 
ALLOWED_EXTENSIONS = {".py", ".md", ".rst", ".txt", ".cfg", ".toml", ".yml", ".yaml"}

BINARY_EXTENSIONS = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".svg", ".ico", ".bmp",
    ".zip", ".tar", ".gz", ".whl", ".egg", ".DS_Store",
    ".pdf", ".docx", ".xlsx", ".mp4", ".mp3",
}

MAX_CHUNK_LINES = 80
MIN_CHUNK_LINES = 3
CHUNK_OVERLAP_LINES = 5

#  Agent Config 
MAX_TOOL_CALLS_PER_QUERY = 6
MAX_ITERATIONS = 10

#  Feature Flags 
# Toggle retrieval critic disable if hitting Groq rate limits during demo
ENABLE_CRITIC = True

#  Tool Config 
SEARCH_DEFAULT_TOP_K = 5
READ_FILE_MAX_LINES = 300
LIST_DIR_MAX_DEPTH = 4
FIND_USAGES_MAX_RESULTS = 20
TRACE_CALL_DEPTH = 2

#  CLI Config 
CLI_SHOW_TRACE = True
CLI_SHOW_TOOLS = True

#  Scope Guard 
REPO_DESCRIPTION = "psf/requests Python HTTP library"
SCOPE_REJECTION_MESSAGE = (
    "This system is scoped exclusively to the `psf/requests` codebase. "
    "I cannot answer questions outside this repository. "
    "Please ask something about the requests library's code, architecture, or API."
)

#  Validation 
def validate_config() -> list[str]:
    """Validate critical config values. Returns list of error messages."""
    errors = []
    if not GROQ_API_KEY:
        errors.append(
            "GROQ_API_KEY is not set. "
            "Please add it to the .env file. "
            "Free key at https://console.groq.com"
        )
    if not REPO_URL:
        errors.append("REPO_URL is not configured.")
    return errors