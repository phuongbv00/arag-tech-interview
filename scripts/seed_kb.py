"""CLI: Load concepts.json into ChromaDB."""

import argparse
from pathlib import Path

import chromadb
from openai import OpenAI

from src.config import Settings
from src.kb.loader import embed_and_upsert, load_concepts


def main() -> None:
    parser = argparse.ArgumentParser(description="Seed the knowledge base into ChromaDB")
    parser.add_argument(
        "--kb-path",
        type=Path,
        default=Path("data/kb/concepts.json"),
        help="Path to concepts JSON file",
    )
    args = parser.parse_args()

    settings = Settings()
    openai_client = OpenAI(api_key=settings.openai_api_key)

    print(f"Loading concepts from {args.kb_path}...")
    concepts = load_concepts(args.kb_path)
    print(f"Loaded {len(concepts)} concepts")

    print(f"Initializing ChromaDB at {settings.chroma_persist_dir}...")
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_name
    )

    print("Embedding and upserting concepts...")
    embed_and_upsert(concepts, collection, openai_client, settings.embedding_model)
    print(f"Done. Collection '{settings.chroma_collection_name}' has {collection.count()} entries.")


if __name__ == "__main__":
    main()
