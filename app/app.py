import uuid
import streamlit as st
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

# Import your existing backend logic
from graph import build_graph

# Re-use your initial state function
def make_initial_state(user_input: str):
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
    }

# --- Page Config ---
st.set_page_config(page_title="Rezept-Agent", page_icon="🍳")
st.title("🍳 Rezept-Agent mit Speisekammer")
st.markdown("Verwalte deine Speisekammer und generiere Rezepte!")

# --- Session State Initialization ---
# --- Session State Initialization ---
if "thread_id" not in st.session_state:
    st.session_state.thread_id = f"session-{uuid.uuid4().hex[:8]}"
    
    # Add a visual indicator while the graph builds
    with st.spinner("Initialisiere Agenten... (das kann einen Moment dauern)"):
        st.session_state.graph = build_graph()
        
    st.session_state.chat_history = [
        {"role": "ai", "content": "Hallo! Was möchtest du heute kochen oder zur Speisekammer hinzufügen?"}
    ]

config: RunnableConfig = {"configurable": {"thread_id": st.session_state.thread_id}}

# --- Render Chat History ---
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Chat Input & Graph Execution ---
if user_input := st.chat_input("z.B. 'füge Hähnchen, Reis hinzu' oder 'koch mir was Italienisches'"):
    
    # 1. Display and store user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # 2. Execute graph
    with st.chat_message("ai"):
        # We use a placeholder so we can stream/update the AI's response in real-time
        response_placeholder = st.empty()
        ai_response = ""
        
        with st.spinner("Agent denkt nach..."):
            # Check if the graph is currently paused (interrupted for user selection)
            snapshot = st.session_state.graph.get_state(config)
            
            if snapshot.next:
                # We are resuming from a pause (e.g. user selected a recipe)
                st.session_state.graph.update_state(
                    config,
                    {"messages": [HumanMessage(content=user_input)]},
                )
                stream_input = None  # None tells LangGraph to resume
            else:
                # We are starting a brand new request
                stream_input = make_initial_state(user_input)

            # Stream the events from the graph
            for event in st.session_state.graph.stream(stream_input, config, stream_mode="updates"):
                if not isinstance(event, dict):
                    continue
                
                # Parse nodes to find AI messages
                for node_name, updates in event.items():
                    if isinstance(updates, dict) and updates.get("messages"):
                        last_msg = updates["messages"][-1]
                        if hasattr(last_msg, "type") and last_msg.type == "ai" and last_msg.content:
                            ai_response = last_msg.content
                            response_placeholder.markdown(ai_response)

        # 3. Store final AI response in history
        if ai_response:
            st.session_state.chat_history.append({"role": "ai", "content": ai_response})