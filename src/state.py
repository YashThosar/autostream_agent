from typing import Annotated, TypedDict, Optional
from langchain_core.messages import BaseMessage
import operator

class LeadData(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]
    intent: str
    lead_data: LeadData
    lead_captured: bool
    rag_context: str
