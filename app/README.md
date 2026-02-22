# Recipe Agent — LangGraph with Persistent Pantry

## Architecture

```
[START]
   │
   ▼
[load_pantry]  ← reads pantry.json from disk
   │
   ▼
[classify_intent]  ← LLM determines: recipe / add / remove / list
   │
   ├── pantry_add ──► add to pantry.json ──► [END]
   ├── pantry_remove ──► remove from pantry.json ──► [END]
   ├── pantry_list ──► display grouped contents ──► [END]
   │
   ▼  (intent = "recipe")
[parse_input]  ← extract extras, constraints, prefs (pantry already loaded)
   │
   ├── (needs_clarification?) ──► [END]
   │
   ▼
[generate_recipes]  ← 3 candidates using pantry + extras
   │
   ▼
   ⏸ INTERRUPT ── user picks recipe(s) or says "more"
   │
   ▼
[handle_selection]
   │
   ├── (wants more + iteration < 3) ──► [generate_recipes]
   │
   ▼
[generate_grocery_list]  ← diffs pantry vs recipe needs
   │
   ▼
[format_output]  ← full instructions
   │
   ▼
[END]
```

## LangGraph patterns demonstrated

| Pattern | Where |
|---|---|
| **Intent-based routing** | `classify_intent` → conditional edges to 4 branches |
| **Persistent external state** | `pantry.json` loaded/saved outside graph state |
| **Conditional edges** | `route_by_intent`, `route_after_parse`, `route_after_selection` |
| **Human-in-the-loop** | `interrupt_before=["handle_selection"]` |
| **Loop with cap** | `iteration_count < 3` on recipe regeneration |
| **Checkpointing** | `MemorySaver` for state across interrupts |

## File structure

```
recipe_agent/
├── state.py          # TypedDict state schema
├── pantry.py         # Pantry CRUD — reads/writes pantry.json
├── nodes.py          # All node functions (LLM calls + pantry ops)
├── graph.py          # Graph wiring, edges, interrupt config
├── run.py            # Interactive REPL
├── requirements.txt
└── pantry.json       # Auto-created on first add (gitignore this)
```

## Setup

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
python run.py
```

## Example session

```
You> add chicken breast, rice, broccoli, soy sauce, garlic, eggs, onion, olive oil
🤖 Added to pantry: chicken breast, rice, broccoli, soy sauce, garlic, eggs, onion, olive oil. You now have 8 items.

You> show pantry
🤖 Your pantry:
   Protein: chicken breast (some)
   Vegetable: broccoli (some), garlic (some), onion (some)
   Grain: rice (some)
   Condiment: soy sauce (some), olive oil (some)
   Dairy: eggs (some)

You> quick asian dinner for 2
🤖 Using your pantry (8 items). Generating recipes...
🤖 Here are your options:
   1. Chicken Fried Rice — ...
   2. Garlic Chicken Stir-Fry — ...
   3. Egg Drop Soup with Rice — ...

You> 1
🤖 Going with: Chicken Fried Rice
🤖 You have everything you need — no shopping required.
🤖 ## Chicken Fried Rice ...
```

## Swapping LLMs

In `nodes.py`, change the `llm` line:

```python
# Anthropic
from langchain_anthropic import ChatAnthropic
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

# Ollama (local, free)
from langchain_ollama import ChatOllama
llm = ChatOllama(model="llama3.1")
```

## Key design decision: pantry outside graph state

The pantry lives in a JSON file, not in LangGraph's checkpointer. Why:
- Pantry persists across all sessions (graph threads are per-conversation)
- Simple to manually edit, backup, or sync
- Graph state stays focused on the current request

If you later want multi-user support or more complex querying, swap `pantry.py` for SQLite — the interface stays the same.
