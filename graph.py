"""LangGraph definition for the recipe agent with persistent pantry."""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import RecipeState
from nodes import (
    load_pantry_node,
    classify_intent,
    pantry_add_node,
    pantry_remove_node,
    pantry_list_node,
    parse_input,
    generate_recipes,
    search_db_recipes,
    handle_selection,
    generate_grocery_list,
    format_output,
)


# ── Routing functions ────────────────────────────────────────────────────────

def route_by_intent(state: RecipeState) -> str:
    """Route to the correct subgraph based on classified intent."""
    intent = state.get("intent", "recipe")
    return {
        "recipe":         "parse_input",
        "recipe_db":      "parse_input",
        "pantry_add":     "pantry_add",
        "pantry_remove":  "pantry_remove",
        "pantry_list":    "pantry_list",
    }.get(intent, "parse_input")


def route_after_parse(state: RecipeState) -> str:
    if state.get("intent") == "recipe_db":
        return "search_db_recipes"  # never block DB search on vague input
    if state.get("needs_clarification"):
        return END
    return "generate_recipes"


def route_after_recipes(state: RecipeState) -> str:
    """Skip handle_selection if there are no candidates to choose from."""
    if not state.get("candidate_recipes"):
        return END
    return "handle_selection"


def route_after_selection(state: RecipeState) -> str:
    if not state.get("user_approved") and state.get("iteration_count", 0) < 3:
        return "generate_recipes"
    return "generate_grocery_list"


# ── Build the graph ──────────────────────────────────────────────────────────

def build_graph():
    builder = StateGraph(RecipeState)

    # ── Nodes ──
    builder.add_node("load_pantry", load_pantry_node)
    builder.add_node("classify_intent", classify_intent)

    # Pantry management nodes
    builder.add_node("pantry_add", pantry_add_node)
    builder.add_node("pantry_remove", pantry_remove_node)
    builder.add_node("pantry_list", pantry_list_node)

    # Recipe flow nodes
    builder.add_node("parse_input", parse_input)
    builder.add_node("generate_recipes", generate_recipes)
    builder.add_node("search_db_recipes", search_db_recipes)
    builder.add_node("handle_selection", handle_selection)
    builder.add_node("generate_grocery_list", generate_grocery_list)
    builder.add_node("format_output", format_output)

    # ── Edges ──

    # Always start: load pantry → classify intent
    builder.add_edge(START, "load_pantry")
    builder.add_edge("load_pantry", "classify_intent")

    # Branch based on intent
    builder.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "parse_input": "parse_input",
            "pantry_add": "pantry_add",
            "pantry_remove": "pantry_remove",
            "pantry_list": "pantry_list",
        },
    )

    # Pantry management → END (single-turn operations)
    builder.add_edge("pantry_add", END)
    builder.add_edge("pantry_remove", END)
    builder.add_edge("pantry_list", END)

    # Recipe flow
    builder.add_conditional_edges(
        "parse_input",
        route_after_parse,
        {
            END:                 END,
            "generate_recipes":  "generate_recipes",
            "search_db_recipes": "search_db_recipes",
        },
    )

    builder.add_conditional_edges(
        "generate_recipes",
        route_after_recipes,
        {END: END, "handle_selection": "handle_selection"},
    )
    builder.add_conditional_edges(
        "search_db_recipes",
        route_after_recipes,
        {END: END, "handle_selection": "handle_selection"},
    )

    builder.add_conditional_edges(
        "handle_selection",
        route_after_selection,
        {
            "generate_recipes": "generate_recipes",
            "generate_grocery_list": "generate_grocery_list",
        },
    )

    builder.add_edge("generate_grocery_list", "format_output")
    builder.add_edge("format_output", END)

    # Compile with checkpointer + interrupt for recipe selection
    memory = MemorySaver()
    graph = builder.compile(
        checkpointer=memory,
        interrupt_before=["handle_selection"],
    )

    return graph


# ── Visualize ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    graph = build_graph()
    try:
        print(graph.get_graph().draw_mermaid())
    except Exception:
        print("Graph built successfully.")
