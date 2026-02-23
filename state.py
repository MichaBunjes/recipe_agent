"""State definition for the Recipe Agent."""

from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class RecipeState(TypedDict):
    """Full state schema for the recipe planning graph."""

    # --- Input ---
    messages: Annotated[list[BaseMessage], add_messages]
    raw_input: str
    intent: str  # "recipe" | "pantry_add" | "pantry_remove" | "pantry_list"

    # --- Pantry (loaded from disk) ---
    pantry_ingredients: list[str]  # flat list from persistent pantry

    # --- Parsed from current request ---
    extra_ingredients: list[str]  # anything mentioned ad-hoc ("I also have lemons today")
    required_ingredients: list[str]  # ingredients the user explicitly requires ("must use garlic")
    dietary_constraints: list[str]
    preferences: dict

    # --- Generated ---
    candidate_recipes: list[dict]
    selected_recipes: list[dict]
    meal_plan: dict
    grocery_list: list[dict]

    # --- Control flow ---
    needs_clarification: bool
    user_approved: bool
    iteration_count: int
    restart_flow: bool  # set by handle_selection when input is not a recipe selection
