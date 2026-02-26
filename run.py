"""
Interactive REPL for the ReAct recipe agent.

One persistent thread per session — conversation history carries over between messages.
Pantry persists in pantry.json regardless of thread.

Usage:
    export OPENAI_API_KEY=sk-...
    python run.py
    python run.py --verbose
"""

import sys
import uuid
from dotenv import load_dotenv

load_dotenv()  # must be first — loads LANGSMITH_* / LANGFUSE_* vars before LangChain initialises

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from graph import build_graph

try:
    from langfuse.langchain import CallbackHandler as LangfuseCallback
    _langfuse_handler = LangfuseCallback()
except Exception:
    _langfuse_handler = None

VERBOSE = "--verbose" in sys.argv or "-v" in sys.argv


def print_event(event: dict) -> None:
    """Print AI messages from a graph event. In verbose mode also show tool calls."""
    if not isinstance(event, dict):
        return
    for node_name, updates in event.items():
        if node_name == "__interrupt__":
            continue
        if not isinstance(updates, dict):
            continue
        messages = updates.get("messages", [])
        for msg in messages:
            if VERBOSE and hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"\n  [tool_call] {tc['name']}({tc['args']})")
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                print(f"\n🤖 {msg.content}\n")


def stream_response(graph, payload, config: RunnableConfig) -> None:
    """Stream graph events and print messages."""
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        print_event(chunk)


def run() -> None:
    graph = build_graph()

    print("=" * 60)
    print("  Rezept-Agent (ReAct)")
    print("=" * 60)
    print()
    print("Befehle:")
    print("  • Hinzufügen:    'füge Hähnchen, Reis, Knoblauch hinzu'")
    print("  • Entfernen:     'entferne Eier'")
    print("  • Speisekammer:  'zeige Speisekammer'")
    print("  • Rezepte (KI):  'koch mir was Italienisches'")
    print("  • Rezepte (DB):  'suche Rezepte in meinen Büchern'")
    print("  • Beenden:       'exit'")
    if VERBOSE:
        print("  [Ausführlicher Modus aktiv]")
    print()

    # One thread per REPL session — history carries over
    thread_id = f"session-{uuid.uuid4().hex[:8]}"
    config: RunnableConfig = {
        "configurable": {"thread_id": thread_id},
        "callbacks": [_langfuse_handler] if _langfuse_handler else [],
    }

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTschüss!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("beenden", "quit", "exit", "q"):
            print("Tschüss!")
            break

        stream_response(
            graph,
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )


if __name__ == "__main__":
    run()
