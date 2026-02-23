"""
Interactive REPL for the recipe agent.

Supports two modes in a single loop:
  - Pantry management: "add chicken, rice, soy sauce" / "remove eggs" / "show pantry"
  - Recipe generation: "make me something italian" / "quick dinner for 2"

Usage:
    pip install -r requirements.txt
    export OPENAI_API_KEY=sk-...
    python run.py
"""

import sys
import uuid
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from graph import build_graph
from state import RecipeState

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def make_initial_state(user_input: str) -> RecipeState:
    return {
        "raw_input": user_input,
        "messages": [HumanMessage(content=user_input)],
        "intent": "",
        "pantry_ingredients": [],
        "extra_ingredients": [],
        "required_ingredients": [],
        "dietary_constraints": [],
        "preferences": {},
        "candidate_recipes": [],
        "selected_recipes": [],
        "meal_plan": {},
        "grocery_list": [],
        "needs_clarification": False,
        "user_approved": False,
        "iteration_count": 0,
        "restart_flow": False,
    }


def print_event(event):
    """Print node updates. In verbose mode also show state changes."""
    if not isinstance(event, dict):
        return
    for node_name, updates in event.items():
        if not isinstance(updates, dict):
            continue
        if VERBOSE:
            print(f"\n  [{node_name}]")
            for key, val in updates.items():
                if key == "messages":
                    continue
                print(f"    {key}: {val!r}")
        if updates.get("messages"):
            last = updates["messages"][-1]
            if hasattr(last, "type") and last.type == "ai" and last.content:
                print(f"\n🤖 {last.content}\n")


def run():
    graph = build_graph()

    print("=" * 60)
    print("  Rezept-Agent mit persistenter Speisekammer")
    print("=" * 60)
    print()
    print("Befehle:")
    print("  • Hinzufügen:    'füge Hähnchen, Reis, Knoblauch hinzu'")
    print("  • Entfernen:     'entferne Eier'")
    print("  • Speisekammer:  'zeige Speisekammer' / 'was habe ich'")
    print("  • Rezepte:       'koch mir was Italienisches' / 'schnelles Abendessen für 2'")
    print("  • Rezept-DB:     'suche passende Rezepte in meinen Büchern'")
    print("  • Beenden:       'beenden' / 'exit'")
    if VERBOSE:
        print("  [Ausführlicher Modus aktiv]")
    print()

    while True:
        user_input = input("You> ").strip()
        if not user_input:
            continue
        if user_input.lower() in ("beenden", "quit", "exit", "q"):
            print("Tschüss!")
            break

        # Each top-level request gets a fresh thread
        # (the pantry persists via JSON file, not via LangGraph state)
        thread_id = f"session-{uuid.uuid4().hex[:8]}"
        config: RunnableConfig = {"configurable": {"thread_id": thread_id}}

        state = make_initial_state(user_input)

        # Run graph until it hits an interrupt or finishes
        for event in graph.stream(state, config, stream_mode="updates"):
            print_event(event)

        # ── Handle interrupt loop (recipe selection) ─────────────────────
        while True:
            snapshot = graph.get_state(config)
            if not snapshot.next:
                break  # Graph finished

            # We're paused at handle_selection — get user's choice
            choice = input("You> ").strip()
            if not choice:
                continue

            graph.update_state(
                config,
                {"messages": [HumanMessage(content=choice)]},
            )

            for event in graph.stream(None, config, stream_mode="updates"):
                print_event(event)


if __name__ == "__main__":
    run()
