import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "rag.env")
load_dotenv(env_path)

# llm
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b-cloud")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# embedding
EMBEDDING_ENDPOINT = os.getenv("EMBEDDING_ENDPOINT", "http://127.0.0.1:11434")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6b")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY")
