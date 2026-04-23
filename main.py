"""
main.py – CLI interface for the AutoStream Sales Agent

Run:
    python main.py

The agent retains full conversation memory across turns.
Type 'quit' or 'exit' to end the session.
"""
import sys
import os

# Ensure src/ is importable
sys.path.insert(0, os.path.dirname(__file__))

from langchain_core.messages import HumanMessage
from src.agent import agent_graph
from src.state import AgentState


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------

def create_initial_state() -> AgentState:
    return {
        "messages": [],
        "intent": "greeting",
        "lead_data": {},
        "lead_captured": False,
    }


# ---------------------------------------------------------------------------
# Pretty print helper
# ---------------------------------------------------------------------------

def print_agent_response(state: AgentState) -> None:
    """Extract and print the latest AI response from state."""
    for msg in reversed(state["messages"]):
        # Skip tool messages and human messages — find the last AI text response
        if hasattr(msg, "content") and msg.__class__.__name__ == "AIMessage":
            if msg.content and not (hasattr(msg, "tool_calls") and msg.tool_calls):
                print(f"\n🤖 Alex (AutoStream): {msg.content}\n")
                return


# ---------------------------------------------------------------------------
# Main conversation loop
# ---------------------------------------------------------------------------

def run() -> None:
    print("=" * 60)
    print("  AutoStream Sales Agent — Configurable LLM Backend")
    print("  Type 'quit' or 'exit' to end the session.")
    print("=" * 60)

    state = create_initial_state()

    # Kick off with a greeting from the agent
    state = agent_graph.invoke(
        {**state, "messages": state["messages"] + [HumanMessage(content="Hi")]},
        config={"recursion_limit": 20},
    )
    print_agent_response(state)

    while True:
        try:
            user_input = input("👤 You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSession ended. Goodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("\n🤖 Alex: Thanks for chatting! Feel free to come back anytime. 👋\n")
            break

        # Append user message and invoke the graph
        state = agent_graph.invoke(
            {**state, "messages": state["messages"] + [HumanMessage(content=user_input)]},
            config={"recursion_limit": 20},
        )

        print_agent_response(state)

        # Session complete after lead is captured
        if state.get("lead_captured"):
            print("✅  Lead capture complete. Session will continue if you have more questions.\n")


if __name__ == "__main__":
    run()
