from typing import TypedDict,Annotated, Sequence
from langchain_core.messages import (
    BaseMessage,
)
import operator


# Define the state with messages
class GraphState(TypedDict):
    question: str
    discarded_responses: Annotated[list[dict], operator.add]
    responses: list[dict]
    difficulty: int
    threads: int
    beams: int
    start: bool
    done: bool
    final_response: str
    agent_response: dict
    index: int
    initial_response_agent: str
    revisions: int

class AgentState(TypedDict):
  messages: Annotated[Sequence[BaseMessage], operator.add]
  sender: str
  final_answer: str