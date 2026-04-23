"""
llm_factory.py - Build a chat model from environment configuration.

Set LLM_PROVIDER to one of:
  - groq
  - openai
  - anthropic
  - google
  - openai_compatible
"""

import os


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float, got: {raw}") from exc


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got: {raw}") from exc


def create_llm():
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    temperature = _get_float("LLM_TEMPERATURE", 0.3)
    max_tokens = _get_int("LLM_MAX_TOKENS", 1024)

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
            api_key=os.getenv("OPENAI_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=os.getenv("LLM_MODEL", "claude-3-5-haiku-latest"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(
            model=os.getenv("LLM_MODEL", "gemini-1.5-flash"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=temperature,
            max_output_tokens=max_tokens,
        )

    if provider == "openai_compatible":
        from langchain_openai import ChatOpenAI

        base_url = os.getenv("OPENAI_BASE_URL")
        if not base_url:
            raise ValueError("OPENAI_BASE_URL is required when LLM_PROVIDER=openai_compatible")

        return ChatOpenAI(
            model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
            api_key=os.getenv("OPENAI_COMPAT_API_KEY") or os.getenv("OPENAI_API_KEY"),
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    raise ValueError(
        "Unsupported LLM_PROVIDER. Use one of: groq, openai, anthropic, google, openai_compatible"
    )
