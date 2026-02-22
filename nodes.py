"""Node functions for the recipe agent graph."""

import json
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

load_dotenv()

from state import RecipeState
from rag import search_recipes, get_recipes_by_ingredient_overlap
from pantry import (
    load_pantry,
    save_pantry,
    add_ingredients,
    remove_ingredients,
    list_ingredients,
    list_by_category,
)


# ── LLM setup ────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)


# ── Node: load_pantry_node ───────────────────────────────────────────────────
def load_pantry_node(state: RecipeState) -> dict:
    """Load persistent pantry from disk into state."""
    pantry = load_pantry()
    return {"pantry_ingredients": list_ingredients(pantry)}


# ── Node: classify_intent ────────────────────────────────────────────────────
def classify_intent(state: RecipeState) -> dict:
    """Determine what the user wants: recipe, pantry management, or listing."""

    system = SystemMessage(content="""Klassifiziere die Absicht des Nutzers in genau eine der folgenden Kategorien:
- "recipe"        — er möchte etwas kochen / neue Rezeptvorschläge vom KI erhalten
- "recipe_db"     — er möchte passende Rezepte aus einer Rezeptbuch-Datenbank suchen
- "pantry_add"    — er möchte Zutaten zur Speisekammer hinzufügen
- "pantry_remove" — er möchte Zutaten aus der Speisekammer entfernen
- "pantry_list"   — er möchte sehen, was in seiner Speisekammer ist

Antworte NUR mit dem Intent-String, nichts weiter.""")

    response = llm.invoke([system, HumanMessage(content=state["raw_input"])])
    intent = str(response.content).strip().strip('"').lower()

    if intent not in ("recipe", "recipe_db", "pantry_add", "pantry_remove", "pantry_list"):
        intent = "recipe"  # default

    return {"intent": intent}


# ── Node: pantry_add ─────────────────────────────────────────────────────────
def pantry_add_node(state: RecipeState) -> dict:
    """Parse and add ingredients to persistent pantry."""

    system = SystemMessage(content="""Extrahiere die Zutaten, die der Nutzer zur Speisekammer hinzufügen möchte.
Übersetze den Zutatennamen ins britische Englisch (z.B. "Eier" → "eggs", "Aubergine" → "aubergine", "Zucchini" → "courgette", "Paprika" → "pepper").
Gib NUR ein JSON-Array von Objekten zurück: [{"name": "eggs", "quantity": "12", "category": "dairy"}, ...]
Kategorien: protein, dairy, vegetable, fruit, grain, spice, condiment, other.
Keine Markdown-Umrandungen.""")

    response = llm.invoke([system, HumanMessage(content=state["raw_input"])])

    try:
        items = json.loads(str(response.content).strip())
    except json.JSONDecodeError:
        return {"messages": [AIMessage(content="Konnte das nicht verarbeiten. Versuche: 'füge Eier, Butter, Reis hinzu'")]}

    pantry = load_pantry()
    pantry = add_ingredients(pantry, items)
    save_pantry(pantry)

    names = [i["name"] for i in items]
    return {
        "pantry_ingredients": list_ingredients(pantry),
        "messages": [AIMessage(content=f"Zur Speisekammer hinzugefügt: {', '.join(names)}. "
                                       f"Du hast jetzt {len(pantry['ingredients'])} Artikel.")],
    }


# ── Node: pantry_remove ─────────────────────────────────────────────────────
def pantry_remove_node(state: RecipeState) -> dict:
    """Parse and remove ingredients from persistent pantry."""

    system = SystemMessage(content="""Extrahiere die Zutatennamen, die der Nutzer aus der Speisekammer entfernen möchte.
Gib NUR ein JSON-Array von Strings zurück: ["Eier", "Butter"]
Keine Markdown-Umrandungen.""")

    response = llm.invoke([system, HumanMessage(content=state["raw_input"])])

    try:
        names = json.loads(str(response.content).strip())
    except json.JSONDecodeError:
        return {"messages": [AIMessage(content="Konnte das nicht verarbeiten. Versuche: 'entferne Eier, Butter'")]}

    pantry = load_pantry()
    pantry = remove_ingredients(pantry, names)
    save_pantry(pantry)

    return {
        "pantry_ingredients": list_ingredients(pantry),
        "messages": [AIMessage(content=f"Entfernt: {', '.join(names)}. "
                                       f"Noch {len(pantry['ingredients'])} Artikel übrig.")],
    }


# ── Node: pantry_list ────────────────────────────────────────────────────────
def pantry_list_node(state: RecipeState) -> dict:
    """Show current pantry contents grouped by category."""
    pantry = load_pantry()
    grouped = list_by_category(pantry)

    if not grouped:
        return {"messages": [AIMessage(content="Deine Speisekammer ist leer. Sage 'füge hinzu ...' um sie zu befüllen.")]}

    lines = []
    for cat, items in sorted(grouped.items()):
        lines.append(f"**{cat.title()}:** {', '.join(items)}")

    return {"messages": [AIMessage(content="Deine Speisekammer:\n" + "\n".join(lines))]}


# ── Node: parse_input ────────────────────────────────────────────────────────
def parse_input(state: RecipeState) -> dict:
    """Parse recipe request. Pantry is already loaded — only extract extras + constraints."""

    pantry_str = ", ".join(state["pantry_ingredients"]) or "leer"

    is_db = state.get("intent") == "recipe_db"
    db_hint = (
        "\nWICHTIG (Rezeptbuch-Suche): Zutaten, die der Nutzer mit 'mit', 'mit ...', "
        "'enthält', 'muss ... haben' nennt, sind IMMER required_ingredients — "
        "sie filtern, welche Rezepte gefunden werden. "
        "extra_ingredients ist NUR für 'ich habe heute auch ...' (ad-hoc Vorrat)."
    ) if is_db else ""

    system = SystemMessage(content=f"""Du bist ein Kochassistent. Die Speisekammer des Nutzers enthält bereits: {pantry_str}
{db_hint}
Analysiere die Rezeptanfrage und extrahiere:
1. extra_ingredients: Zutaten, die der Nutzer ad-hoc als VORRAT erwähnt ("ich habe heute auch Basilikum") — Übersetze den Zutatennamen ins britische Englisch (z.B. "Eier" → "eggs", "Aubergine" → "aubergine", "Zucchini" → "courgette", "Paprika" → "pepper")
2. required_ingredients: Zutaten, die das Rezept ENTHALTEN MUSS ("mit Aubergine", "Knoblauch muss rein") — Übersetze den Zutatennamen ins britische Englisch (z.B. "Eier" → "eggs", "Aubergine" → "aubergine", "Zucchini" → "courgette", "Paprika" → "pepper") \
   , egal ob bereits in der Speisekammer oder nicht
3. dietary_constraints: Liste der Ernährungseinschränkungen
4. preferences: Dict mit optionalen Schlüsseln: cuisine, max_cook_time (Minuten), servings, difficulty
5. needs_clarification: bool — WICHTIG: Setze dies IMMER auf false, wenn die Speisekammer nicht leer ist. \
Setze es nur auf true, wenn die Speisekammer leer UND die Anfrage völlig unverständlich ist.

Antworte NUR mit gültigem JSON, keine Markdown-Umrandungen. Beispiel:
{{
    "extra_ingredients": ["fresh basil"],
    "required_ingredients": ["aubergine"],
    "dietary_constraints": ["vegetarisch"],
    "preferences": {{"cuisine": "italienisch", "servings": 2}},
    "needs_clarification": false,
    "clarification_question": null
}}""")

    response = llm.invoke([system, HumanMessage(content=state["raw_input"])])

    try:
        parsed = json.loads(str(response.content).strip())
    except json.JSONDecodeError:
        return {
            "needs_clarification": True,
            "messages": [AIMessage(content="Konnte das nicht verarbeiten. Was möchtest du kochen?")],
        }

    needs_clarification = parsed.get("needs_clarification", False) and not is_db

    if is_db:
        messages = []  # search_db_recipes will produce the first message
    elif needs_clarification:
        messages = [AIMessage(content=parsed.get("clarification_question") or "Was möchtest du kochen?")]
    else:
        extra = parsed.get("extra_ingredients", [])
        messages = [AIMessage(
            content=f"Nutze deine Speisekammer ({len(state['pantry_ingredients'])} Artikel)"
                    + (f" + {extra}" if extra else "")
                    + ". Rezepte werden generiert..."
        )]

    return {
        "extra_ingredients": parsed.get("extra_ingredients", []),
        "required_ingredients": parsed.get("required_ingredients", []),
        "dietary_constraints": parsed.get("dietary_constraints", []),
        "preferences": parsed.get("preferences", {}),
        "needs_clarification": needs_clarification,
        "messages": messages,
    }


# ── Node: generate_recipes ───────────────────────────────────────────────────
def generate_recipes(state: RecipeState) -> dict:
    """Generate 3 candidate recipes using pantry + extras."""

    all_ingredients = state["pantry_ingredients"] + state.get("extra_ingredients", [])
    ingredients_str = ", ".join(all_ingredients)
    constraints = ", ".join(state["dietary_constraints"]) or "keine"
    prefs = json.dumps(state["preferences"], ensure_ascii=False) if state["preferences"] else "keine"
    required = state.get("required_ingredients", [])
    required_str = ", ".join(required) if required else "keine"

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

    user_msg = f"""Verfügbare Zutaten: {ingredients_str}
Pflicht-Zutaten (müssen in ALLEN 3 Rezepten vorkommen): {required_str}
Ernährungseinschränkungen: {constraints}
Präferenzen: {prefs}"""

    response = llm.invoke([system, HumanMessage(content=user_msg)])

    try:
        recipes = json.loads(str(response.content).strip())
    except json.JSONDecodeError:
        recipes = []

    iteration = state.get("iteration_count", 0) + 1

    summary_parts = []
    for i, r in enumerate(recipes, 1):
        extra = f" | Einkaufen: {', '.join(r['ingredients_needed'])}" if r.get("ingredients_needed") else ""
        summary_parts.append(
            f"{i}. **{r['name']}** — {r['description']} "
            f"({r['cook_time_minutes']} Min., {r['difficulty']}{extra})"
        )
    summary = "Hier sind deine Optionen:\n\n" + "\n".join(summary_parts)
    summary += "\n\nWähle eine aus (z.B. '1'), mehrere ('1 und 3') oder 'mehr' für andere Vorschläge."

    return {
        "candidate_recipes": recipes,
        "iteration_count": iteration,
        "messages": [AIMessage(content=summary)],
    }


# ── Node: handle_selection ───────────────────────────────────────────────────
def handle_selection(state: RecipeState) -> dict:
    """Process user's recipe selection."""
    last_msg = state["messages"][-1].content.lower()
    candidates = state["candidate_recipes"]

    if any(w in last_msg for w in ("mehr", "andere", "different", "more", "other")):
        return {"user_approved": False}

    selected = []
    for i, recipe in enumerate(candidates):
        if str(i + 1) in last_msg or recipe["name"].lower() in last_msg:
            selected.append(recipe)

    if not selected and candidates:
        selected = [candidates[0]]

    return {
        "selected_recipes": selected,
        "user_approved": True,
        "messages": [AIMessage(content=f"Gut, ich nehme: {', '.join(r['name'] for r in selected)}")],
    }


# ── Node: generate_grocery_list ──────────────────────────────────────────────
def generate_grocery_list(state: RecipeState) -> dict:
    """Diff pantry + extras vs recipe needs."""
    have = [i.lower() for i in state["pantry_ingredients"] + state.get("extra_ingredients", [])]

    def _have_it(item: str) -> bool:
        item_lower = item.lower()
        return any(p in item_lower or item_lower in p for p in have)

    grocery = []

    for recipe in state["selected_recipes"]:
        for item in recipe.get("ingredients_needed", []):
            if not _have_it(item):
                grocery.append({"item": item, "for_recipe": recipe["name"]})

    if grocery:
        lines = [f"- {g['item']} (für {g['for_recipe']})" for g in grocery]
        msg = "🛒 **Einkaufsliste:**\n" + "\n".join(lines)
    else:
        msg = "Du hast alles, was du brauchst — kein Einkaufen nötig."

    return {
        "grocery_list": grocery,
        "messages": [AIMessage(content=msg)],
    }


# ── Node: search_db_recipes ──────────────────────────────────────────────────
def search_db_recipes(state: RecipeState) -> dict:
    """Retrieve best matching recipes from ChromaDB and normalise to candidate_recipes schema."""
    print("Durchsuche Rezeptbücher...", flush=True)

    all_ingredients = state["pantry_ingredients"] + state.get("extra_ingredients", [])
    ingredients_str = ", ".join(all_ingredients) or "beliebige Zutaten"
    constraints = ", ".join(state.get("dietary_constraints", [])) or "keine"
    prefs = json.dumps(state.get("preferences", {}), ensure_ascii=False) or "keine"

    query = (
        f"Rezept mit: {ingredients_str}. "
        f"Einschränkungen: {constraints}. "
        f"Präferenzen: {prefs}."
    )

    required = state.get("required_ingredients", [])

    if all_ingredients:
        # Primary: full metadata scan ranked by pantry ingredient overlap
        # Hard-filter to required ingredients first, then rank by total overlap
        hits = get_recipes_by_ingredient_overlap(all_ingredients, n_results=5, required=required)
    else:
        # Fallback: vector search when pantry is empty (pure semantic query)
        hits = search_recipes(query, n_results=5)

    if not hits:
        return {
            "candidate_recipes": [],
            "messages": [AIMessage(
                content="Die Rezeptbuch-Datenbank ist leer. "
                        "Bitte zuerst 'python ingest.py' ausführen."
            )],
        }

    chunks_block = "\n\n---\n\n".join(
        f"[Quelle: {h['metadata'].get('source_file', '?')}]\n{h['text']}"
        for h in hits
    )

    system = SystemMessage(content="""Du bist ein Kochassistent. Dir werden mehrere Rezepttexte aus Rezeptbüchern gegeben (getrennt durch ---).
Extrahiere JEDES Rezept als separaten Eintrag. Wichtige Regeln:
- Extrahiere ALLE Rezepte, nicht nur das erste
- Verwende NUR Informationen, die explizit im Text stehen
- Erfinde KEINE Zutaten oder Schritte, die nicht im Text erwähnt sind

Für jedes Rezept:
- name: string (Rezeptname aus dem Text)
- source: string (der Dateiname aus dem [Quelle: ...] Tag des jeweiligen Abschnitts)
- description: 1 Satz auf Deutsch (aus dem Text)
- ingredients_used: Zutaten aus dem Text, die der Nutzer bereits hat
- ingredients_needed: Zutaten aus dem Text, die noch fehlen
- cook_time_minutes: int oder null
- difficulty: "einfach" | "mittel" | "schwer" oder null
- steps: Zubereitungsschritte aus dem Text (kann leer sein)

Antworte NUR mit einem JSON-Array. Keine Markdown-Umrandungen.""")

    user_msg = (
        f"Verfügbare Zutaten des Nutzers: {ingredients_str}\n\n"
        f"Rezepttexte:\n{chunks_block}"
    )

    response = llm.invoke([system, HumanMessage(content=user_msg)])

    try:
        recipes = json.loads(str(response.content).strip())
        if not isinstance(recipes, list):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        recipes = []

    if not recipes:
        return {
            "candidate_recipes": [],
            "messages": [AIMessage(
                content="Konnte keine passenden Rezepte aus der Datenbank extrahieren. "
                        "Versuche 'koch mir etwas' für KI-generierte Vorschläge."
            )],
        }

    summary_parts = []
    for i, r in enumerate(recipes, 1):
        meta_parts = []
        if r.get("cook_time_minutes"):
            meta_parts.append(f"{r['cook_time_minutes']} Min.")
        if r.get("difficulty"):
            meta_parts.append(r["difficulty"])
        if r.get("ingredients_needed"):
            meta_parts.append(f"Einkaufen: {', '.join(r['ingredients_needed'])}")
        meta = f" ({', '.join(meta_parts)})" if meta_parts else ""
        source = f" _{r['source']}_" if r.get("source") else ""
        summary_parts.append(f"{i}. **{r['name']}**{source} — {r.get('description', '')}{meta}")
    summary = "Hier sind die passendsten Rezepte aus deinen Rezeptbüchern:\n\n"
    summary += "\n".join(summary_parts)
    summary += "\n\nWähle eine aus (z.B. '1'), mehrere ('1 und 3') oder 'mehr' für neue Suche."

    return {
        "candidate_recipes": recipes,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [AIMessage(content=summary)],
    }


# ── Node: format_output ─────────────────────────────────────────────────────
def format_output(state: RecipeState) -> dict:
    """Full recipe instructions."""
    parts = []
    for recipe in state["selected_recipes"]:
        steps = "\n".join(f"  {i}. {s}" for i, s in enumerate(recipe.get("steps", []), 1))
        time_str = f"⏱ {recipe['cook_time_minutes']} Min. | " if recipe.get("cook_time_minutes") else ""
        diff_str = f"Schwierigkeit: {recipe['difficulty']}" if recipe.get("difficulty") else ""
        meta_line = (time_str + diff_str).strip(" |")
        source_line = f"_{recipe['source']}_\n" if recipe.get("source") else ""
        body = (
            f"## {recipe['name']}\n"
            + source_line
            + f"{recipe.get('description', '')}\n"
            + (f"{meta_line}\n" if meta_line else "")
        )
        if steps:
            body += f"\n**Zubereitung:**\n{steps}"
        parts.append(body)

    final = "\n\n---\n\n".join(parts)
    return {"messages": [AIMessage(content=final)]}
