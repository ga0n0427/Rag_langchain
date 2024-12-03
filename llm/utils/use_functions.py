from typing import List
from langchain.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
)
import textwrap

def replace_t(texts):
    """
    문자열 리스트의 각 항목에서 탭(\t)을 공백으로 대체합니다.

    Args:
        texts (list of str): 텍스트 문자열 리스트.

    Returns:
        list of str: 탭이 공백으로 대체된 텍스트 리스트.
    """
    return [text.replace('\t', ' ') for text in texts]

def replace_t_with_space(list_of_documents):
    """
    줄 바꿈 공백으로 대체하는 함수
    """
    for doc in list_of_documents:
        doc.page_content = doc.page_content.replace('\t', ' ')  # Replace tabs with spaces
    return list_of_documents

def text_wrap(text, width=120):
    """
    텍스트 wrapping함수
    """
    return textwrap.fill(text, width=width)

def format_docs(documents):
    """
    문서 목록을 하나의 문자열로 포맷팅합니다.

    Args:
    documents (list): Document 객체 리스트

    Returns:
    str: 포맷된 문자열, 각 문서는 <facts> 태그 안에 포함됨
    """
    formatted_docs = "\n".join([f"<facts>{doc.page_content}</facts>" for doc in documents])
    return formatted_docs

# 테스트
def preprocess_spacing_and_whitespace(query):
    """
    공백 제거를 수행합니다.
    
    Args:
        query (str): 입력 쿼리문.
    
    Returns:
        str: 공백이 제거된 쿼리문.
    """
    # 1. 앞뒤 공백 제거
    query = query.strip()
    
    return query

