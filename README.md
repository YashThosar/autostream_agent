# AutoStream Sales Agent
### Social-to-Lead Agentic Workflow — ServiceHive / Inflx Assignment

A production-grade conversational AI agent that qualifies leads for **AutoStream** (a fictional SaaS video editing platform). Built with **LangGraph**, a **provider-agnostic LLM layer** (Groq/OpenAI/Anthropic/Gemini/OpenAI-compatible), and a local **RAG pipeline**.

## Demo Video

- Repository file: [demo/demo-video.mp4](demo/demo-video.mp4)
- Direct GitHub link: [Watch demo video](https://github.com/YashThosar/autostream_agent/blob/main/demo/demo-video.mp4)

---

## Features

- **Intent Detection** — classifies every user message into: `greeting`, `product_inquiry`, or `high_intent`
- **RAG-Powered Q&A** — answers pricing and policy questions using a local TF-IDF knowledge base
- **Stateful Lead Collection** — collects name → email → platform one field at a time, never skipping ahead
- **Safe Tool Execution** — `capture_lead` only fires after all three fields are confirmed
- **Memory Across Turns** — full conversation history is retained across 5–6+ turns via LangGraph state

---

## Project Structure

```
autostream_agent/
├── data/
│   └── knowledge_base.json     # Local KB: pricing, features, policies
├── src/
│   ├── state.py                # AgentState TypedDict (LangGraph state schema)
│   ├── rag.py                  # TF-IDF retrieval pipeline
│   ├── tools.py                # LangChain tools: search_knowledge_base, capture_lead
│   ├── llm_factory.py          # Provider-agnostic LLM initializer
│   └── agent.py                # LangGraph StateGraph + nodes + routing
├── main.py                     # CLI entry point
├── requirements.txt
├── .env.example
└── README.md
```

---

## How to Run Locally

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/autostream-agent.git
cd autostream-agent
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure your LLM provider
```bash
cp .env.example .env
# Open .env and set:
#   LLM_PROVIDER (groq/openai/anthropic/google/openai_compatible)
#   LLM_MODEL
#   matching API key for your provider
```

### 5. Run the agent
```bash
python main.py
```

### Example Session
```
🤖 Alex: Hi! I'm Alex from AutoStream. We help content creators automate their video editing...

👤 You: What plans do you offer?
🤖 Alex: We have two plans — Basic at $29/month (10 videos, 720p) and Pro at $79/month (unlimited, 4K, AI captions).

👤 You: That sounds great, I want to try the Pro plan for my YouTube channel.
🤖 Alex: Awesome! Let me get you set up. What's your name?

👤 You: Rahul Sharma
🤖 Alex: Great, Rahul! What's your email address?

👤 You: rahul@example.com
🤖 Alex: Perfect. Which platform do you primarily create on?

👤 You: YouTube
✅  Lead captured: Rahul Sharma, rahul@example.com, YouTube
🤖 Alex: You're all set, Rahul! Our team will reach out shortly with onboarding details.
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because it offers **explicit, inspectable state management** via a typed `StateGraph`. In this project, each turn follows a deterministic path: `rag` classifies intent + retrieves context, then routing sends flow to `lead_capture` (for high-intent/ongoing lead collection) or directly to `agent` for response generation. This keeps behavior transparent, testable, and easy to debug. AutoGen is stronger for multi-agent collaboration; LangGraph is a better fit for this **single-agent workflow with controlled routing**.

### How State is Managed

A `AgentState` TypedDict is passed through every node. It holds:

- **`messages`** — the full conversation history, auto-appended via `operator.add`. This gives the configured LLM complete context on every invocation, enabling true multi-turn memory.
- **`intent`** — updated after each user message using lightweight rule-based classification (complementing LLM reasoning).
- **`lead_data`** — accumulates `name`, `email`, and `platform` as the user provides them.
- **`lead_captured`** — a boolean flag set to `True` when `capture_lead` successfully fires.

Conversation memory is preserved by passing prior `messages` back into each graph invocation and appending the latest `HumanMessage`. This gives the agent reliable multi-turn context for intent tracking and lead-field extraction.

### RAG Pipeline

The local knowledge base (`knowledge_base.json`) is indexed using **TF-IDF vectorization** (scikit-learn). On each query, cosine similarity retrieves the top-2 most relevant chunks, which are injected into the LLM context. This is fully local — no embedding API calls required.

---

## WhatsApp Deployment via Webhooks

### Overview

To deploy this agent on WhatsApp, the approach is:

1. **WhatsApp Business API** (Meta Cloud API or a BSP like Twilio) is configured to send incoming messages to a webhook URL via HTTP POST.
2. A **FastAPI server** (or Flask) exposes a `POST /webhook` endpoint.
3. The endpoint extracts the user's message and `phone_number_id`, runs it through the LangGraph agent, and sends the response back via the WhatsApp Send Message API.

### Session Management

Since WhatsApp is stateless, each phone number needs its own persisted `AgentState`. This is handled by storing state in **Redis** (or PostgreSQL) keyed by `phone_number`:

```python
state = redis.get(phone_number) or create_initial_state()
state = agent_graph.invoke({
    **state,
    "messages": state["messages"] + [HumanMessage(user_text)]
})
redis.set(phone_number, state)
```

### Webhook Flow

```
WhatsApp User
     │
     ▼ (sends message)
Meta Cloud API
     │
     ▼ POST /webhook
FastAPI Server
     │
     ├── Verify webhook token (GET /webhook)
     ├── Extract message + sender
     ├── Load session state from Redis
     ├── Run agent_graph.invoke(...)
     ├── Save updated state to Redis
     └── POST reply → Meta Send Message API
     │
     ▼
WhatsApp User receives response
```

### Key Code Sketch

```python
# webhook.py (FastAPI)
from fastapi import FastAPI, Request
import httpx, json, os

app = FastAPI()
VERIFY_TOKEN = os.getenv("WHATSAPP_VERIFY_TOKEN")
WA_TOKEN     = os.getenv("WHATSAPP_ACCESS_TOKEN")

@app.get("/webhook")
async def verify(hub_mode: str, hub_verify_token: str, hub_challenge: str):
    if hub_mode == "subscribe" and hub_verify_token == VERIFY_TOKEN:
        return int(hub_challenge)
    return {"error": "Forbidden"}, 403

@app.post("/webhook")
async def receive_message(request: Request):
    body     = await request.json()
    entry    = body["entry"][0]["changes"][0]["value"]
    message  = entry["messages"][0]
    sender   = message["from"]
    text     = message["text"]["body"]
    phone_id = entry["metadata"]["phone_number_id"]

    # Load / update session state
    state = load_state(sender) or create_initial_state()
    state = agent_graph.invoke({
        **state,
        "messages": state["messages"] + [HumanMessage(text)]
    })
    save_state(sender, state)

    # Extract and send reply
    reply = get_last_ai_text(state)
    await send_whatsapp_message(phone_id, sender, reply, WA_TOKEN)
    return {"status": "ok"}
```

### Infrastructure for Production

| Component | Tool |
|-----------|------|
| Webhook server | FastAPI on Railway / Render / EC2 |
| Session store | Redis (Upstash for serverless) |
| WhatsApp API | Meta Cloud API (free tier) |
| SSL (required by Meta) | Let's Encrypt / Cloudflare |

---

## Evaluation Checklist

| Criterion | How it's addressed |
|-----------|-------------------|
| Intent detection | Rule-based classifier + LLM reasoning via system prompt |
| RAG | TF-IDF retriever over local JSON KB, injected into LLM context |
| State management | LangGraph `AgentState` with `operator.add` for messages |
| Tool calling | `mock_lead_capture` fires only after all 3 required fields are collected |
| Code clarity | Modular structure: state / rag / tools / agent / main |
| Deployability | WhatsApp webhook design documented above |

---

## Tech Stack

- **Python 3.10+**
- **LangGraph 0.2+** — state graph orchestration
- **LangChain 0.3+** — LLM interface, tool definitions
- **Configurable LLM Provider** — Groq / OpenAI / Anthropic / Gemini / OpenAI-compatible
- **scikit-learn** — TF-IDF RAG retrieval
- **python-dotenv** — environment config
