from langchain.schema import Document
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain.schema import Document
from typing import List
import cohere
from dotenv import load_dotenv
import os
from config import TOP_K_RESULTS
load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY")
cohere_client = cohere.Client(COHERE_API_KEY)

def bm25_search(documents: List[Document], query: str, top_n: int = TOP_K_RESULTS) -> List[Document]:
    """
    BM25로 문서를 검색하고 재정렬합니다.
    :param documents: 문서 리스트 (Document 객체).
    :param query: 검색 질의(Query).
    :param top_n: 반환할 문서 개수 (기본값: 10).
    :return: 재정렬된 문서 리스트.
    """
    try:
        # BM25 검색기 초기화
        retriever = BM25Retriever.from_documents(documents)

        # 검색 결과 얻기
        results = retriever.invoke(query)[:top_n]
        return results
    except Exception as e:
        # 예외 발생 시 오류 출력 및 빈 리스트 반환
        print(f"Error during BM25 search: {e}")
        return []


def search_and_rerank(query: str, vectorstore, top_n: int = TOP_K_RESULTS) -> List[Document]:
    """
    FAISS로 검색한 뒤, BM25로 재정렬하는 전체 워크플로.
    :param query: 검색 질의(Query).
    :param vectorstore: 벡터 스토어 객체 (FAISS).
    :param top_n: 반환할 문서 개수 (기본값: 3).
    :return: 재정렬된 문서 리스트.
    """
    try:
        # Step 1: FAISS에서 유사도 검색
        if hasattr(vectorstore, 'similarity_search'):
            search_results = vectorstore.similarity_search(query, k=10)
        else:
            raise ValueError("vectorstore는 FAISS 인스턴스여야 합니다.")
        
        # Step 2: BM25로 재정렬
        retriever = BM25Retriever.from_documents(search_results)
        reranked_results = retriever.invoke(query)[:top_n]

        # 유효한 문서만 필터링
        valid_documents = [doc for doc in reranked_results if doc.page_content.strip()]
        return valid_documents
    except Exception as e:
        # 예외 발생 시 오류 출력 및 빈 리스트 반환
        print(f"Error during search and rerank: {e}")
        return []


def search_with_faiss(query, vectorstore, top_n):
    """
    FAISS를 이용해 문서를 검색합니다.
    :param query: 검색 질의(Query).
    :param vectorstore: 벡터 스토어 객체 (FAISS).
    :param top_n: 반환할 문서 개수.
    :return: 검색된 문서 리스트.
    """
    try:
        # FAISS에서 유사도 검색 수행
        results = vectorstore.similarity_search(query, k=top_n)
        return results
    except Exception as e:
        # 예외 발생 시 오류 출력 및 빈 리스트 반환
        print(f"Error during FAISS search: {e}")
        return []


def cohere_rerank_sim(documents: List[Document], query: str, top_n: int = TOP_K_RESULTS) -> Optional[List[Document]]:
    """
    Cohere를 사용하여 문서를 재정렬합니다.
    :param documents: 문서 리스트 (Document 객체).
    :param query: 검색 질의(Query).
    :param top_n: 반환할 문서 개수 (기본값: 10).
    :return: 재정렬된 문서 리스트 또는 None (모든 점수가 0.9 이하인 경우).
    """
    try:
        # 문서 콘텐츠 추출
        contents = [doc.page_content for doc in documents]

        # Cohere rerank 호출
        response = cohere_client.rerank(
            query=query,
            documents=contents,
            top_n=len(documents),  # 검색된 모든 문서를 대상으로 재정렬
            model="rerank-multilingual-v3.0"  # 적절한 모델 선택
        )

        # Cohere 응답에서 점수와 문서를 결합
        ranked_docs = [
            (documents[result.index], result.relevance_score)
            for result in response.results
        ]

        # 점수를 기준으로 정렬
        ranked_docs.sort(key=lambda x: x[1], reverse=True)

        # 반환할 문서 개수 제한
        return [doc for doc, _ in ranked_docs[:top_n]]
    except Exception as e:
        # 예외 발생 시 오류 출력 및 None 반환
        print(f"Error during Cohere reranking: {e}")
        return None


def search_and_cohere_rerank(query: str, vectorstore, top_n: int = TOP_K_RESULTS) -> List[Document]:
    """
    FAISS로 검색한 뒤, Cohere로 재정렬하는 전체 워크플로.
    :param query: 검색 질의(Query).
    :param vectorstore: 벡터 스토어 객체 (FAISS).
    :param top_n: 반환할 문서 개수 (기본값: 3).
    :return: 재정렬된 문서 리스트.
    """
    try:
        # Step 1: FAISS에서 유사도 검색
        if hasattr(vectorstore, 'similarity_search'):
            search_results = vectorstore.similarity_search(query, k=30)  # 검색 단계는 10개 고정
        else:
            raise ValueError("vectorstore는 FAISS 인스턴스여야 합니다.")
        
        # Step 2: Cohere로 재정렬
        reranked_results = cohere_rerank_sim(search_results, query, top_n=top_n)

        # 재정렬 결과가 None인 경우 빈 리스트 반환
        if reranked_results is None:
            return []

        return reranked_results
    except Exception as e:
        # 예외 발생 시 오류 출력 및 빈 리스트 반환
        print(f"Error during search and Cohere rerank: {e}")
        return []
    
def cohere_rerank(documents: List[Document], query: str, top_n: int = TOP_K_RESULTS) -> Optional[List[Document]]:
    """
    Cohere를 사용하여 문서를 재정렬합니다.
    :param documents: 문서 리스트 (Document 객체).
    :param query: 검색 질의(Query).
    :param top_n: 반환할 문서 개수 (기본값: 10).
    """
    try:
        # 문서 콘텐츠 추출
        contents = [doc.page_content for doc in documents]

        # Cohere rerank 호출
        response = cohere_client.rerank(
            query=query,
            documents=contents,
            top_n=len(documents),  # 검색된 모든 문서를 대상으로 재정렬
            model="rerank-multilingual-v3.0"  # 적절한 모델 선택
        )

        # Cohere 응답에서 점수와 문서를 결합
        ranked_docs = [
            (documents[result.index], result.relevance_score)
            for result in response.results
        ]
        
        # 점수를 기준으로 정렬
        ranked_docs.sort(key=lambda x: x[1], reverse=True)
        # 반환할 문서 개수 제한
        return [doc for doc, _ in ranked_docs[:top_n]]
    except Exception as e:
        # 예외 발생 시 오류 출력 및 None 반환
        print(f"Error during Cohere reranking: {e}")
        return None


def cohere_rerank_only(query: str, vectorstore, top_n: int = TOP_K_RESULTS) -> List[Document]:
    """
    Vectorstore에 저장된 모든 문서를 Cohere로 재정렬하는 함수.
    :param query: 검색 질의(Query).
    :param vectorstore: FAISS 벡터스토어 객체.
    :param top_n: 반환할 문서 개수 (기본값: 3).
    :return: 재정렬된 문서 리스트.
    """
    try:
        # Step 1: Vectorstore에서 모든 문서 가져오기
        documents = list(vectorstore.docstore._dict.values())  # InMemoryDocstore의 문서 접근 방식
        if not documents:
            print("Vectorstore에 문서가 없습니다.")
            return []

        # Step 2: Cohere로 문서 재정렬
        reranked_results = cohere_rerank(documents, query, top_n=top_n)

        # 재정렬 결과가 None인 경우 빈 리스트 반환
        if reranked_results is None:
            return []

        return reranked_results
    except Exception as e:
        # 예외 발생 시 오류 출력 및 빈 리스트 반환
        print(f"Error during Cohere rerank: {e}")
        return []
    