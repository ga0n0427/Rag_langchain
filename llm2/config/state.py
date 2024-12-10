from typing import TypedDict, Annotated, Sequence
from langchain.schema import Document
import operator

# 그래프의 상태를 정의하는 클래스

class MyState(TypedDict):
    context: list
    answer: str
    question: str
    query_evaluated: float
    query_feedback: str
"""
class MyState(TypedDict):
    context: Annotated[Sequence[Document], operator.add]
    answer: Annotated[Sequence[str], operator.add]
    question: Annotated[str, operator.add]
    query_evaluated: float
    query_feedback: Annotated[str, operator.add]
    """