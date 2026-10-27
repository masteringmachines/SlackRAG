import os

from dotenv import load_dotenv

load_dotenv()

SLACK_BOT_TOKEN = os.environ["SLACK_BOT_TOKEN"]
SLACK_APP_TOKEN = os.environ["SLACK_APP_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")

ALLOWED_USER_IDS = {x.strip() for x in os.getenv("ALLOWED_USER_IDS", "").split(",") if x.strip()}

MAX_AGENT_STEPS = 6
HISTORY_TURNS = 8
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 5
MAX_FILE_BYTES = 25 * 1024 * 1024
SUPPORTED_EXTS = {".txt", ".md", ".pdf", ".csv", ".json", ".log", ".py"}
