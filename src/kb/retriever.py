from __future__ import annotations

import json

import chromadb
from openai import OpenAI

from src.kb.schema import ConceptEntry


class KBRetriever:
    """Wrapper around ChromaDB collection for domain-specific queries."""

    def __init__(
        self,
        collection: chromadb.Collection,
        openai_client: OpenAI,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.collection = collection
        self.openai_client = openai_client
        self.embedding_model = embedding_model

    def _embed(self, text: str) -> list[float]:
        response = self.openai_client.embeddings.create(
            input=[text], model=self.embedding_model
        )
        return response.data[0].embedding

    def _metadata_to_concept(self, metadata: dict) -> ConceptEntry:
        """Convert ChromaDB metadata dict back to a ConceptEntry."""
        parsed = {}
        for key, value in metadata.items():
            if key in ("key_properties", "common_misconceptions", "related_concepts"):
                parsed[key] = json.loads(value) if isinstance(value, str) else value
            else:
                parsed[key] = value
        return ConceptEntry.model_validate(parsed)

    def get_by_concept_id(self, concept_id: str) -> ConceptEntry | None:
        """Direct lookup by concept_id (metadata filter, no embedding needed)."""
        results = self.collection.get(ids=[concept_id], include=["metadatas"])
        if not results["metadatas"]:
            return None
        return self._metadata_to_concept(results["metadatas"][0])

    def search_by_response(
        self, response_text: str, top_k: int = 5
    ) -> list[tuple[ConceptEntry, float]]:
        """Dense vector search using response text as query.

        Returns list of (concept, similarity_score) tuples.
        """
        query_embedding = self._embed(response_text)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "distances"],
        )

        output: list[tuple[ConceptEntry, float]] = []
        if results["metadatas"] and results["distances"]:
            for metadata, distance in zip(
                results["metadatas"][0], results["distances"][0]
            ):
                concept = self._metadata_to_concept(metadata)
                similarity = 1.0 - distance  # ChromaDB returns L2 distance by default
                output.append((concept, similarity))
        return output

    def get_related_concepts(self, concept_id: str) -> list[ConceptEntry]:
        """Retrieve all related concepts by following related_concepts links."""
        concept = self.get_by_concept_id(concept_id)
        if concept is None:
            return []

        related: list[ConceptEntry] = []
        for related_id in concept.related_concepts:
            related_concept = self.get_by_concept_id(related_id)
            if related_concept is not None:
                related.append(related_concept)
        return related

    def get_by_difficulty(
        self,
        difficulty: str,
        exclude_ids: list[str] | None = None,
    ) -> list[ConceptEntry]:
        """Filter concepts by difficulty level, optionally excluding already-covered ones."""
        results = self.collection.get(
            where={"difficulty_level": difficulty},
            include=["metadatas"],
        )

        concepts: list[ConceptEntry] = []
        exclude_set = set(exclude_ids) if exclude_ids else set()
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                concept = self._metadata_to_concept(metadata)
                if concept.concept_id not in exclude_set:
                    concepts.append(concept)
        return concepts

    def get_all_concepts(self) -> list[ConceptEntry]:
        """Retrieve all concepts from the collection."""
        results = self.collection.get(include=["metadatas"])
        concepts: list[ConceptEntry] = []
        if results["metadatas"]:
            for metadata in results["metadatas"]:
                concepts.append(self._metadata_to_concept(metadata))
        return concepts
