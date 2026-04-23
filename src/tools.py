"""
tools.py – LangChain tools available to the AutoStream Sales Agent.

Two tools:
  1. search_knowledge_base  – RAG retrieval over local KB
  2. capture_lead           – Mock lead capture API (fires ONLY when all fields collected)
"""
from langchain_core.tools import tool
from src.rag import retrieve


# ---------------------------------------------------------------------------
# Tool 1 : Knowledge Base Search (RAG)
# ---------------------------------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """
    Search the AutoStream knowledge base to answer questions about pricing,
    plans, features, and company policies.

    Use this tool whenever the user asks about:
    - Pricing (Basic plan, Pro plan, costs)
    - Features (4K, AI captions, video limits)
    - Policies (refunds, support availability)

    Args:
        query: The user's question or topic to search for.

    Returns:
        Relevant information retrieved from the knowledge base.
    """
    return retrieve(query)


# ---------------------------------------------------------------------------
# Tool 2 : Lead Capture (Mock API)
# ---------------------------------------------------------------------------

def mock_lead_capture(name: str, email: str, platform: str) -> None:
    """
    Mock backend API that registers a qualified lead.
    In production this would POST to a CRM (HubSpot, Salesforce, etc.)
    """
    print(f"\n{'='*50}")
    print(f"✅  Lead captured successfully!")
    print(f"    Name     : {name}")
    print(f"    Email    : {email}")
    print(f"    Platform : {platform}")
    print(f"{'='*50}\n")


@tool
def capture_lead(name: str, email: str, platform: str) -> str:
    """
    Capture a qualified lead after collecting all required information.

    ⚠️  IMPORTANT: Only call this tool AFTER you have confirmed all three values:
        - name    (the user's full name)
        - email   (a valid email address)
        - platform (their creator platform: YouTube, Instagram, TikTok, etc.)

    DO NOT call this with placeholder or missing values.

    Args:
        name     : Full name of the lead.
        email    : Email address of the lead.
        platform : The creator platform they use (YouTube, Instagram, TikTok, etc.)

    Returns:
        Confirmation message to relay to the user.
    """
    # Call the mock backend API
    mock_lead_capture(name, email, platform)

    return (
        f"Lead successfully registered for {name} ({email}) on {platform}. "
        f"Our team will reach out shortly with onboarding details!"
    )
