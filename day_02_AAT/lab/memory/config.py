import os
from dotenv import load_dotenv

current_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(current_dir, "memory.env")
load_dotenv(env_path)

# llm
LLM_ENDPOINT = os.getenv("LLM_ENDPOINT", "http://127.0.0.1:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-oss:20b-cloud")
LLM_API_KEY = os.getenv("LLM_API_KEY")

# langsmith
LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "lgcns")
