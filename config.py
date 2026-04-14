import os
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")

APP_ENV = os.getenv("APP_ENV", "development")
PORT = int(os.getenv("PORT", "8000"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

DATABASE_DIR = Path(os.getenv("DATABASE_DIR", str(ROOT_DIR / "data")))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", str(ROOT_DIR / "output")))

CSV_PATH = ROOT_DIR / "processed_data.csv"  # 파싱v1
PKL_PATH_V21 = OUTPUT_DIR  / "processed_data_v21.pkl"  # 파싱v2(표 포함)
PKL_PATH_V22 = OUTPUT_DIR  / "processed_data_v22.pkl"  # 파싱v2(표 미포함)

FAISS_INDEX_DIR = ROOT_DIR / "faiss_index"
FAISS_INDEX_DIR_V2 = ROOT_DIR / "faiss_index_v2"

EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(f"Missing required environment variable: {name}")
    return value


OPENAI_API_KEY = require_env("OPENAI_API_KEY")