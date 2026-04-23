"""
agent.py - LangGraph-powered AutoStream Sales Agent.
"""

import os
import re
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.llm_factory import create_llm
from src.rag import retrieve
from src.tools import mock_lead_capture

load_dotenv()

_LLM = create_llm()

AUTOSTREAM_KB = """
AutoStream Pricing & Plans:
- Basic Plan: $29/month — 10 videos/month, 720p resolution
- Pro Plan: $79/month — Unlimited videos, 4K resolution, AI captions

Company Policies:
- No refunds after 7 days of purchase
- 24/7 support is available ONLY on the Pro plan
- Basic plan users get standard business-hours support
"""

# ---------------------------------------------------------------------------
# Intent classification
# ---------------------------------------------------------------------------

def _classify_intent(messages: list, current_intent: str, lead_data: dict, lead_captured: bool) -> str:
    if lead_captured:
        return "product_inquiry"

    # 1. Already in collection and incomplete → stay in lead_collection
    if lead_data and not all([lead_data.get("name"), lead_data.get("email"), lead_data.get("platform")]):
        return "lead_collection"

    if current_intent == "lead_collection":
        return "lead_collection"

    # 2. Check if the last AI message was asking for lead info
    last_ai = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    lead_ask_phrases = ["your name", "your email", "creator platform", "which platform",
                        "sign you up", "get you started", "tell me your name",
                        "what's your name", "can i get your"]
    if any(p in last_ai.lower() for p in lead_ask_phrases):
        return "lead_collection"

    # 3. Check latest user message for high intent
    last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    text_lower = last_user.lower()

    HIGH_INTENT = ["sign up", "signup", "want to try", "i want the", "purchase",
                   "buy", "subscribe", "get started", "i'm interested", "let's go",
                   "sounds good", "i'll take", "i want pro", "enroll", "ready to",
                   "wants to try", "wants pro", "want pro", "try pro", "yes", "sure",
                   "okay", "ok", "let's do it", "pro plan"]

    if any(s in text_lower for s in HIGH_INTENT):
        # Only high_intent if previous AI message was about a plan/signup
        if any(p in last_ai.lower() for p in ["pro plan", "sign up", "get started",
                                                "would you like", "shall we", "plan"]):
            return "high_intent"

    # Pure high intent keywords regardless of context
    STRONG_HIGH_INTENT = ["sign up", "signup", "purchase", "buy", "subscribe",
                           "want to try", "wants to try", "want pro", "wants pro"]
    if any(s in text_lower for s in STRONG_HIGH_INTENT):
        return "high_intent"

    if len(last_user.split()) <= 3 and text_lower.strip("!.") in ["hi", "hello", "hey"]:
        return "greeting"

    return "product_inquiry"


# ---------------------------------------------------------------------------
# Node 1: rag_node
# ---------------------------------------------------------------------------

def rag_node(state: AgentState) -> dict:
    messages       = state["messages"]
    lead_data      = state.get("lead_data", {})
    current_intent = state.get("intent", "greeting")
    lead_captured  = state.get("lead_captured", False)

    last_user = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    intent    = _classify_intent(messages, current_intent, lead_data, lead_captured)
    retrieved = retrieve(last_user)
    context   = AUTOSTREAM_KB + "\n\nRelevant details:\n" + retrieved

    return {
        "intent": intent,
        "rag_context": context,
        "messages": [],
        "lead_data": lead_data,
        "lead_captured": state.get("lead_captured", False),
    }


# ---------------------------------------------------------------------------
# Node 2: lead_capture_node
# ---------------------------------------------------------------------------

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}")
PLATFORMS  = ["youtube", "instagram", "tiktok", "twitter", "facebook",
              "linkedin", "twitch", "snapchat", "pinterest", "threads"]

def _extract_email(text):
    m = _EMAIL_RE.search(text)
    return m.group(0) if m else None

def _extract_platform(text):
    lower = text.lower()
    for p in PLATFORMS:
        if p in lower:
            return p.capitalize()
    return None


def _extract_name(text: str):
    value = text.strip()
    if not value:
        return None

    patterns = [
        r"^(?:my name is|i am|i'm)\s+([a-zA-Z][a-zA-Z .'-]{1,60})$",
    ]
    for pattern in patterns:
        m = re.match(pattern, value, flags=re.IGNORECASE)
        if m:
            return m.group(1).strip().title()

    if 1 <= len(value.split()) <= 5 and re.fullmatch(r"[a-zA-Z .'-]+", value):
        return value.title()

    return None

def lead_capture_node(state: AgentState) -> dict:
    messages  = state["messages"]
    lead_data = dict(state.get("lead_data", {}))
    intent    = state.get("intent", "")

    if intent not in ("high_intent", "lead_collection"):
        return {
            "messages": [], "lead_data": lead_data,
            "lead_captured": state.get("lead_captured", False),
            "intent": intent, "rag_context": state.get("rag_context", ""),
        }

    human_msgs = [m.content for m in messages if isinstance(m, HumanMessage)]
    last_ai    = next((m.content for m in reversed(messages) if isinstance(m, AIMessage)), "")
    last_human = human_msgs[-1].strip() if human_msgs else ""

    # Extract email
    if not lead_data.get("email"):
        email = _extract_email(last_human)
        if email:
            lead_data["email"] = email

    # Extract platform
    if not lead_data.get("platform"):
        platform = _extract_platform(last_human)
        if platform:
            lead_data["platform"] = platform

    # Extract name — only if AI just asked for name
    name_asked = any(p in last_ai.lower() for p in
                     ["your name", "tell me your name", "what's your name",
                      "may i have your name", "can i get your name"])

    if (name_asked
            and not lead_data.get("name")
            and not _extract_email(last_human)
            and not _extract_platform(last_human)
            and not any(k in last_human.lower() for k in
                        ["plan", "pro", "basic", "want", "sign", "yes", "no",
                         "okay", "ok", "sure", "hi", "hello", "price"])):
        extracted_name = _extract_name(last_human)
        if extracted_name:
            lead_data["name"] = extracted_name

    lead_captured = state.get("lead_captured", False)
    if (not lead_captured
            and lead_data.get("name")
            and lead_data.get("email")
            and lead_data.get("platform")):
        mock_lead_capture(lead_data["name"], lead_data["email"], lead_data["platform"])
        lead_captured = True

    return {
        "messages": [],
        "lead_data": lead_data,
        "lead_captured": lead_captured,
        "intent": "product_inquiry" if lead_captured else "lead_collection",
        "rag_context": state.get("rag_context", ""),
    }


# ---------------------------------------------------------------------------
# Node 3: agent_node
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are Alex, a friendly AI sales assistant for AutoStream — a SaaS platform for automated video editing.

## AutoStream Knowledge Base
{kb}

## Lead Collection Status
{lead_status}

## Instructions
1. Greet users warmly when they say hello.
2. Answer pricing/policy questions using the knowledge base above.
3. When a user shows interest in a plan (says "pro plan", "want to try", "sign up", "yes", "sure", etc.) → immediately start collecting details.
4. Collect ONE field at a time in order:
   - Name missing → ask ONLY "What's your name?"
   - Name done, email missing → ask ONLY "What's your email address?"
   - Name + email done, platform missing → ask ONLY "Which platform do you create on? (YouTube, Instagram, etc.)"
   - All done → thank them and confirm registration
5. NEVER ask for a field already collected. NEVER ask for multiple fields at once.
6. If lead_status says "ask for X next" — ask for X immediately, nothing else.

## Tone: Friendly, concise, no fluff.
"""

def agent_node(state: AgentState) -> dict:
    rag_context   = state.get("rag_context", AUTOSTREAM_KB)
    lead_data     = state.get("lead_data", {})
    lead_captured = state.get("lead_captured", False)

    collected = [f"{f}: {lead_data[f]}" for f in ["name","email","platform"] if lead_data.get(f)]
    missing   = [f for f in ["name","email","platform"] if not lead_data.get(f)]

    if lead_captured:
        lead_status = "All fields collected and lead registered. Thank the user warmly."
    elif collected:
        lead_status = (f"Collected so far: {', '.join(collected)}. "
                       f"Ask for '{missing[0]}' next. Do not ask for anything else.")
    else:
        lead_status = "Nothing collected yet. If user shows interest in a plan, ask for their name."

    filled_prompt = SYSTEM_PROMPT.format(kb=rag_context, lead_status=lead_status)
    system        = SystemMessage(content=filled_prompt)
    response      = _LLM.invoke([system] + state["messages"])

    return {
        "messages": [response],
        "intent": state.get("intent", "product_inquiry"),
        "lead_data": lead_data,
        "lead_captured": lead_captured,
        "rag_context": rag_context,
    }


# ---------------------------------------------------------------------------
# Routing + Graph
# ---------------------------------------------------------------------------

def route_after_rag(state: AgentState) -> Literal["lead_capture", "agent"]:
    if state.get("lead_captured"):
        return "agent"
    if state.get("intent") in ("high_intent", "lead_collection"):
        return "lead_capture"
    return "agent"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("rag",          rag_node)
    graph.add_node("lead_capture", lead_capture_node)
    graph.add_node("agent",        agent_node)
    graph.set_entry_point("rag")
    graph.add_conditional_edges("rag", route_after_rag,
                                 {"lead_capture": "lead_capture", "agent": "agent"})
    graph.add_edge("lead_capture", "agent")
    graph.add_edge("agent",        END)
    return graph.compile()

agent_graph = build_graph()