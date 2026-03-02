from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings


def _load_yaml(filename: str) -> dict:
    config_dir = Path(__file__).resolve().parent.parent / "configs"
    path = config_dir / filename
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f) or {}
    return {}


_defaults = _load_yaml("default.yaml")
_llm = _defaults.get("llm", {})
_embedding = _defaults.get("embedding", {})
_chroma = _defaults.get("chroma", {})
_session = _defaults.get("session", {})
_retrieval = _defaults.get("retrieval", {})


class Settings(BaseSettings):
    """Application settings loaded from .env + configs/default.yaml."""

    # API keys (from .env)
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    # LLM
    llm_model: str = _llm.get("model", "gpt-4o")
    llm_temperature: float = _llm.get("temperature", 0.3)
    llm_max_tokens: int = _llm.get("max_tokens", 2048)

    # Embedding
    embedding_model: str = _embedding.get("model", "text-embedding-3-small")

    # ChromaDB
    chroma_persist_dir: str = _chroma.get("persist_dir", "./chroma_db")
    chroma_collection_name: str = _chroma.get("collection_name", "dsa_concepts")

    # Session
    max_topics: int = _session.get("max_topics", 5)
    max_depth_per_topic: int = _session.get("max_depth_per_topic", 3)
    max_total_turns: int = _session.get("max_total_turns", 30)

    # Retrieval
    retrieval_top_k: int = _retrieval.get("top_k", 5)
    similarity_threshold: float = _retrieval.get("similarity_threshold", 0.7)

    model_config = {"env_file": ".env", "extra": "ignore"}
