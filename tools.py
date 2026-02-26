"""LangChain tools for the ReAct recipe agent."""

import json
from typing import Optional
from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

from pantry import (
    load_pantry,
    save_pantry,
    add_ingredients,
    remove_ingredients,
    list_ingredients,
    list_by_category,
)
from rag import search_recipes, get_recipes_by_ingredient_overlap

_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── Pantry tools ─────────────────────────────────────────────────────────────

@tool
def get_pantry() -> str:
    """Lade und zeige den aktuellen Inhalt der Speisekammer als flache Liste."""
    pantry = load_pantry()
    items = list_ingredients(pantry)
    if not items:
        return "Die Speisekammer ist leer."
    return f"Speisekammer ({len(items)} Artikel): {', '.join(items)}"


@tool
def list_pantry_by_category() -> str:
    """Zeige den Inhalt der Speisekammer nach Kategorie gruppiert."""
    pantry = load_pantry()
    grouped = list_by_category(pantry)
    if not grouped:
        return "Die Speisekammer ist leer."
    lines = [f"**{cat.title()}:** {', '.join(items)}" for cat, items in sorted(grouped.items())]
    return "Deine Speisekammer:\n" + "\n".join(lines)


@tool
def add_to_pantry(items: list[dict]) -> str:
    """Füge Zutaten zur Speisekammer hinzu.

    items: Liste von Objekten mit den Feldern:
      - name (str, auf britischem Englisch, z.B. "eggs", "aubergine", "courgette")
      - quantity (str, optional, z.B. "500g", "1 Packung") — Standard: "some"
      - category (str, optional) — Standard: "other"
        Kategorien: protein, dairy, vegetable, fruit, grain, spice, condiment, other
    """
    pantry = load_pantry()
    pantry = add_ingredients(pantry, items)
    save_pantry(pantry)
    names = [i["name"] for i in items]
    return (
        f"Zur Speisekammer hinzugefügt: {', '.join(names)}. "
        f"Du hast jetzt {len(pantry['ingredients'])} Artikel."
    )


@tool
def remove_from_pantry(names: list[str]) -> str:
    """Entferne Zutaten aus der Speisekammer.

    names: Liste von Zutatennamen in der exakten Schreibweise der Speisekammer
           (z.B. ["broccoli", "eggs"]). Verwende get_pantry() wenn unsicher.
    """
    pantry = load_pantry()
    pantry = remove_ingredients(pantry, names)
    save_pantry(pantry)
    return (
        f"Entfernt: {', '.join(names)}. "
        f"Noch {len(pantry['ingredients'])} Artikel übrig."
    )


# ── Recipe tools ─────────────────────────────────────────────────────────────

@tool
def generate_ai_recipes(
    extra_ingredients: Optional[list[str]] = None,
    required_ingredients: Optional[list[str]] = None,
    dietary_constraints: Optional[list[str]] = None,
    cuisine: str = "",
    servings: int = 0,
    max_cook_time_minutes: int = 0,
) -> str:
    """Generiere 3 KI-Rezeptvorschläge basierend auf der Speisekammer.

    extra_ingredients: ad-hoc Zutaten, die der Nutzer heute extra hat (z.B. ["fresh basil"])
    required_ingredients: Zutaten, die in ALLEN Rezepten vorkommen müssen (z.B. ["aubergine"])
    dietary_constraints: Ernährungseinschränkungen (z.B. ["vegetarisch", "glutenfrei"])
    cuisine: Gewünschte Küche (z.B. "asiatisch", "italienisch")
    servings: Anzahl Portionen (0 = beliebig)
    max_cook_time_minutes: Maximale Kochzeit in Minuten (0 = beliebig)

    Gibt ein JSON-Array mit 3 Rezepten zurück.
    """
    pantry = load_pantry()
    pantry_items = list_ingredients(pantry)

    extra = extra_ingredients or []
    required = required_ingredients or []
    dietary = dietary_constraints or []

    all_ingredients = pantry_items + extra
    ingredients_str = ", ".join(all_ingredients) or "keine Zutaten vorhanden"
    constraints_str = ", ".join(dietary) or "keine"
    required_str = ", ".join(required) if required else "keine"

    prefs_parts = []
    if cuisine:
        prefs_parts.append(f"Küche: {cuisine}")
    if servings:
        prefs_parts.append(f"Portionen: {servings}")
    if max_cook_time_minutes:
        prefs_parts.append(f"Max. Kochzeit: {max_cook_time_minutes} Min.")
    prefs_str = ", ".join(prefs_parts) or "keine"

    system = SystemMessage(content="""Du bist ein kreativer Koch. Generiere genau 3 Rezeptvorschläge auf Deutsch.
Priorisiere Rezepte, die hauptsächlich vorhandene Zutaten verwenden. Für jedes Rezept:
- name: string
- description: 1 Satz
- ingredients_used: Liste (aus den vorhandenen Zutaten)
- ingredients_needed: Liste (was noch gekauft werden müsste — so wenig wie möglich)
- cook_time_minutes: int
- difficulty: "einfach" | "mittel" | "schwer"
- steps: Liste von Anweisungen

Antworte NUR mit einem JSON-Array. Keine Markdown-Umrandungen.""")

    user_msg = (
        f"Verfügbare Zutaten: {ingredients_str}\n"
        f"Pflicht-Zutaten (müssen in ALLEN 3 Rezepten vorkommen): {required_str}\n"
        f"Ernährungseinschränkungen: {constraints_str}\n"
        f"Präferenzen: {prefs_str}"
    )

    response = _llm.invoke([system, HumanMessage(content=user_msg)])

    try:
        recipes = json.loads(str(response.content).strip())
    except json.JSONDecodeError:
        return "[]"

    return json.dumps(recipes, ensure_ascii=False)


@tool
def search_cookbook(
    extra_ingredients: Optional[list[str]] = None,
    required_ingredients: Optional[list[str]] = None,
) -> str:
    """Durchsuche die Rezeptbuch-Datenbank nach passenden Rezepten.

    Verwendet vollständigen Zutaten-Overlap-Scan über ALLE Rezepte — kein Rezept wird ausgelassen.
    Rankt nach Übereinstimmung mit der Speisekammer.

    extra_ingredients: ad-hoc Zutaten (z.B. ["basil", "lemon"])
    required_ingredients: Pflicht-Zutaten, die jedes Rezept enthalten muss (z.B. ["aubergine"])

    Gibt ein JSON-Array mit Rezepten zurück.
    """
    pantry = load_pantry()
    pantry_items = list_ingredients(pantry)
    extra = extra_ingredients or []
    required = required_ingredients or []

    all_ingredients = pantry_items + extra

    if all_ingredients:
        hits = get_recipes_by_ingredient_overlap(all_ingredients, n_results=10, required=required)
    else:
        hits = search_recipes("Rezept", n_results=10)

    if not hits:
        return json.dumps({"error": "Keine Rezepte in der Datenbank. Bitte 'python ingest.py' ausführen."})

    chunks_block = "\n\n---\n\n".join(
        f"[Quelle: {h['metadata'].get('source_file', '?')}]\n{h['text']}"
        for h in hits
    )
    ingredients_str = ", ".join(all_ingredients) or "beliebige Zutaten"

    system = SystemMessage(content="""Du bist ein Kochassistent. Dir werden Rezepttexte aus Rezeptbüchern gegeben (getrennt durch ---).
Extrahiere JEDES Rezept als separaten Eintrag. Regeln:
- Extrahiere ALLE Rezepte, nicht nur das erste
- Verwende NUR Informationen aus dem Text
- Erfinde KEINE Zutaten oder Schritte

Für jedes Rezept:
- name: string
- source: string (Dateiname aus dem [Quelle: ...] Tag)
- description: 1 Satz auf Deutsch
- ingredients_used: Zutaten aus dem Text, die der Nutzer bereits hat
- ingredients_needed: Zutaten, die noch fehlen
- cook_time_minutes: int oder null
- difficulty: "einfach" | "mittel" | "schwer" oder null
- steps: Zubereitungsschritte aus dem Text

Antworte NUR mit einem JSON-Array. Keine Markdown-Umrandungen.""")

    user_msg = f"Verfügbare Zutaten des Nutzers: {ingredients_str}\n\nRezepttexte:\n{chunks_block}"
    response = _llm.invoke([system, HumanMessage(content=user_msg)])

    try:
        recipes = json.loads(str(response.content).strip())
        if not isinstance(recipes, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        return "[]"

    return json.dumps(recipes, ensure_ascii=False)



@tool
def get_grocery_list(recipe_name: str, ingredients_needed: list[str]) -> str:
    """Erstelle eine Einkaufsliste für ein gewähltes Rezept.

    recipe_name: Name des Rezepts
    ingredients_needed: Liste der Zutaten, die das Rezept benötigt
                        (aus dem 'ingredients_needed' Feld des gewählten Rezepts)

    Vergleicht mit der Speisekammer und gibt nur wirklich fehlende Zutaten zurück.
    """
    pantry = load_pantry()
    have = [i.lower() for i in list_ingredients(pantry)]

    def _have_it(item: str) -> bool:
        item_lower = item.lower()
        return any(p in item_lower or item_lower in p for p in have)

    missing = [item for item in ingredients_needed if not _have_it(item)]

    if not missing:
        return "Du hast alles, was du brauchst — kein Einkaufen nötig."

    lines = [f"- {item}" for item in missing]
    return f"🛒 **Einkaufsliste für {recipe_name}:**\n" + "\n".join(lines)


# ── Tool registry ─────────────────────────────────────────────────────────────

all_tools = [
    get_pantry,
    list_pantry_by_category,
    add_to_pantry,
    remove_from_pantry,
    generate_ai_recipes,
    search_cookbook,
    get_grocery_list,
]
