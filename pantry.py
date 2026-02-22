"""Persistent pantry storage backed by a local JSON file."""

import json
from pathlib import Path
from datetime import datetime

DEFAULT_PANTRY_PATH = Path(__file__).parent / "pantry.json"


def load_pantry(path: Path = DEFAULT_PANTRY_PATH) -> dict:
    """Load pantry from disk. Returns full pantry dict."""
    if path.exists():
        return json.loads(path.read_text())
    return {"ingredients": {}, "last_updated": None}


def save_pantry(pantry: dict, path: Path = DEFAULT_PANTRY_PATH) -> None:
    """Write pantry to disk."""
    pantry["last_updated"] = datetime.now().isoformat()
    path.write_text(json.dumps(pantry, indent=2))


def add_ingredients(pantry: dict, items: list[dict]) -> dict:
    """Add or update ingredients.

    items: [{"name": "eggs", "quantity": "12", "category": "dairy"}, ...]
    """
    for item in items:
        name = item["name"].lower().strip()
        pantry["ingredients"][name] = {
            "quantity": item.get("quantity", "some"),
            "category": item.get("category", "other"),
            "added": datetime.now().isoformat(),
        }
    return pantry


def remove_ingredients(pantry: dict, names: list[str]) -> dict:
    """Remove ingredients by name."""
    for name in names:
        pantry["ingredients"].pop(name.lower().strip(), None)
    return pantry


def list_ingredients(pantry: dict) -> list[str]:
    """Return flat list of ingredient names."""
    return list(pantry["ingredients"].keys())


def list_by_category(pantry: dict) -> dict[str, list[str]]:
    """Group ingredients by category."""
    grouped: dict[str, list[str]] = {}
    for name, info in pantry["ingredients"].items():
        cat = info.get("category", "other")
        grouped.setdefault(cat, []).append(f"{name} ({info.get('quantity', '?')})")
    return grouped
