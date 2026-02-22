"""ChromaDB client and recipe search utilities."""

from pathlib import Path
import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "recipe_books"

_client = None
_collection = None


def _get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),  # type: ignore[arg-type]
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def search_recipes(query: str, n_results: int = 5) -> list[dict]:
    """Query ChromaDB for the n_results most relevant recipe chunks."""
    collection = _get_collection()
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"] or []
    metas = results["metadatas"] or []
    dists = results["distances"] or []
    return [
        {"text": doc, "metadata": meta, "distance": dist}
        for doc, meta, dist in zip(docs[0], metas[0], dists[0])
    ]


def get_recipes_by_ingredient_overlap(
    pantry: list[str], n_results: int = 5, required: list[str] | None = None
) -> list[dict]:
    """
    Rank ALL recipes by pantry ingredient overlap via metadata scan (no embedding needed).
    Returns top n_results as {text, metadata, overlap} dicts, best first.
    Falls back to an empty list if the collection is empty.
    """
    collection = _get_collection()
    if collection.count() == 0:
        return []

    pantry_lower = [i.lower() for i in pantry]
    all_data = collection.get(include=["metadatas", "documents"])

    def _matches(pantry_item: str, recipe_ings: list[str]) -> bool:
        """True if pantry_item is a substring of any recipe ingredient, or vice versa."""
        return any(
            pantry_item in r or r in pantry_item
            for r in recipe_ings
        )

    required_lower = [r.lower() for r in (required or [])]

    scored = []
    for id_, meta, doc in zip(
        all_data["ids"], all_data["metadatas"] or [], all_data["documents"] or []
    ):
        recipe_ings = [
            ing.strip().lower()
            for ing in str(meta.get("ingredient_names", "")).split(",")
            if len(ing.strip()) >= 3
        ]
        # Hard filter: skip recipes missing any required ingredient
        if required_lower and not all(_matches(r, recipe_ings) for r in required_lower):
            continue
        overlap = sum(1 for p in pantry_lower if _matches(p, recipe_ings))
        scored.append((overlap, id_, meta, doc))

    scored.sort(key=lambda x: -x[0])
    return [
        {"text": doc, "metadata": meta, "overlap": overlap}
        for overlap, _, meta, doc in scored[:n_results]
    ]


def get_collection_for_ingest():
    """Exposed for ingest.py."""
    return _get_collection()
