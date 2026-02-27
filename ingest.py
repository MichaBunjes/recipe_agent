"""
CLI script: ingest PDF recipe books into ChromaDB.

Usage:
    python ingest.py                        # process all PDFs in recipe_books/
    python ingest.py recipe_books/buch.pdf  # single file
    python ingest.py --reset                # wipe collection and re-ingest all
"""

import sys
import json
import hashlib
from pathlib import Path
from pypdf import PdfReader
from db_client import get_collection_for_ingest, COLLECTION_NAME, DefaultEmbeddingFunction

RECIPE_BOOKS_DIR = Path(__file__).parent / "recipe_books"

CHUNK_SIZE = 600
CHUNK_OVERLAP = 100


def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + CHUNK_SIZE].strip())
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if len(c) > 50]


def _extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _chunk_id(source: str, index: int) -> str:
    return hashlib.sha256(f"{source}::{index}".encode()).hexdigest()[:16]


def ingest_json_file(json_path: Path, collection) -> int:
    print(f"  Lese: {json_path.name} ...")
    data = json.loads(json_path.read_text(encoding="utf-8"))
    source = data.get("metadata", {}).get("source", json_path.stem)
    recipes = data.get("recipes", [])
    if not recipes:
        print(f"  Warnung: Keine Rezepte gefunden in {json_path.name}")
        return 0

    ids, documents, metadatas = [], [], []
    for recipe in recipes:
        steps = recipe.get("instructions", [])
        steps_text = "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))
        base = recipe.get("search_text") or (
            f"{recipe.get('title', '')}\n"
            f"Zutaten: {', '.join(recipe.get('ingredient_names', []))}\n"
            f"{recipe.get('description', '')}"
        )
        doc = f"{base}\n\nZubereitung:\n{steps_text}" if steps_text else base
        ids.append(recipe["id"])
        documents.append(doc.strip())
        metadatas.append({
            "source_file": json_path.name,
            "source": source,
            "title": recipe.get("title", ""),
            "chapter": recipe.get("chapter", ""),
            "serves": recipe.get("serves", ""),
            "ingredient_names": ",".join(recipe.get("ingredient_names", [])),
        })

    existing = set(collection.get(ids=ids)["ids"])
    new_idx = [i for i, id_ in enumerate(ids) if id_ not in existing]

    if not new_idx:
        print(f"  Bereits vorhanden, übersprungen: {json_path.name}")
        return 0

    collection.add(
        ids=[ids[i] for i in new_idx],
        documents=[documents[i] for i in new_idx],
        metadatas=[metadatas[i] for i in new_idx],
    )
    print(f"  Hinzugefügt: {len(new_idx)} Rezepte aus {json_path.name}")
    return len(new_idx)


def ingest_file(pdf_path: Path, collection) -> int:
    print(f"  Lese: {pdf_path.name} ...")
    text = _extract_text(pdf_path)
    if not text.strip():
        print(f"  Warnung: Kein Text extrahiert aus {pdf_path.name} (möglicherweise gescannt)")
        return 0

    chunks = _chunk_text(text)
    source_key = pdf_path.stem
    ids = [_chunk_id(source_key, i) for i in range(len(chunks))]
    metadatas = [{"source_file": pdf_path.name, "chunk_index": i} for i in range(len(chunks))]

    existing = set(collection.get(ids=ids)["ids"])
    new_idx = [i for i, id_ in enumerate(ids) if id_ not in existing]

    if not new_idx:
        print(f"  Bereits vorhanden, übersprungen: {pdf_path.name}")
        return 0

    collection.add(
        ids=[ids[i] for i in new_idx],
        documents=[chunks[i] for i in new_idx],
        metadatas=[metadatas[i] for i in new_idx],
    )
    print(f"  Hinzugefügt: {len(new_idx)} neue Chunks aus {pdf_path.name}")
    return len(new_idx)


def main():
    args = sys.argv[1:]
    collection = get_collection_for_ingest()

    if "--reset" in args:
        args = [a for a in args if a != "--reset"]
        from db_client import _client
        _client.delete_collection(COLLECTION_NAME)
        collection = _client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=DefaultEmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )
        print("Kollektion zurückgesetzt.")

    if args:
        all_files = [Path(a) for a in args if Path(a).suffix.lower() in (".pdf", ".json")]
    else:
        all_files = sorted(RECIPE_BOOKS_DIR.glob("*.pdf")) + sorted(RECIPE_BOOKS_DIR.glob("*.json"))

    if not all_files:
        print("Keine PDF- oder JSON-Dateien gefunden. Lege Rezeptbücher in recipe_books/ ab.")
        return

    print(f"\nVerarbeite {len(all_files)} Datei(en)...")

    def ingest_any(path: Path) -> int:
        if path.suffix.lower() == ".json":
            return ingest_json_file(path, collection)
        return ingest_file(path, collection)

    total = sum(ingest_any(f) for f in all_files)
    print(f"\nFertig. {total} neue Chunks gespeichert. Gesamt in DB: {collection.count()} Chunks.")


if __name__ == "__main__":
    main()
