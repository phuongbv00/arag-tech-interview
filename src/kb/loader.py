from __future__ import annotations

import json
from pathlib import Path

import chromadb
from openai import OpenAI

from src.kb.schema import ConceptEntry


def load_concepts(path: Path) -> list[ConceptEntry]:
    """Load and validate all concept entries from a JSON file."""
    with open(path) as f:
        raw = json.load(f)
    return [ConceptEntry.model_validate(entry) for entry in raw]


def _build_embedding_text(concept: ConceptEntry) -> str:
    """Build the text used for embedding a concept.

    Concatenates concept_name, definition, and key_properties.
    """
    properties_text = "; ".join(concept.key_properties)
    return f"{concept.concept_name}\n{concept.definition}\n{properties_text}"


def embed_and_upsert(
    concepts: list[ConceptEntry],
    collection: chromadb.Collection,
    openai_client: OpenAI,
    embedding_model: str = "text-embedding-3-small",
) -> None:
    """Generate embeddings for each concept and upsert into ChromaDB.

    Embedding text = concept_name + definition + key_properties.
    Metadata stores full ConceptEntry fields for filtered retrieval.
    """
    texts = [_build_embedding_text(c) for c in concepts]

    response = openai_client.embeddings.create(input=texts, model=embedding_model)
    embeddings = [item.embedding for item in response.data]

    ids = [c.concept_id for c in concepts]
    metadatas = [c.model_dump() for c in concepts]
    documents = texts

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        metadatas=metadatas,
        documents=documents,
    )
