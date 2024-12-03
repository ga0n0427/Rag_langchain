from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from utils import replace_t_with_space
from typing import List
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from config import FAISS_PATH, TOP_K_RESULTS
import faiss

def check_vectorstore(FAISS_PATH = FAISS_PATH):
    """
    벡터스토어가 존재하는지 확인하고, 없으면 새로 생성합니다.
    """
    try: 
        vectorstore = load_faiss_index(FAISS_PATH)
    except:
        # 빈 벡터스토어 생성 (예: L2 distance, 128차원)
        dimension = 128
        index = faiss.IndexFlatL2(dimension)  # 벡터 간 L2 거리 계산
        # 새로 생성된 벡터스토어 저장
        faiss.write_index(index, FAISS_PATH)
        print("빈 벡터스토어가 생성되었습니다.")

def create_vectorstore(texts, metadatas=None, index_path=FAISS_PATH):
    """
    벡터 스토어를 생성하고 로컬에 저장.

    Args:
        texts (list): 텍스트 청크 리스트.
        metadatas (list): 각 텍스트 청크에 해당하는 메타데이터 리스트.
        index_path (str): 벡터 스토어 저장 경로.

    Returns:
        FAISS: 생성된 벡터 스토어 객체.
    """
    texts = replace_t_with_space(texts)
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vectorstore.save_local(index_path)
    return vectorstore

def build_faiss_index(documents, faiss_path: str):
    """
    주어진 문서들로 FAISS 인덱스를 생성합니다.
    
    :param documents: 문자열 또는 Document 객체의 리스트.
    :param faiss_path: 생성된 FAISS 인덱스를 저장할 경로.
    :return: 생성된 FAISS 인덱스.
    """
    # Ensure `documents` contains valid strings or process `Document` objects
    document_objects = [
        Document(page_content=text.page_content, metadata=text.metadata) 
        if isinstance(text, Document) 
        else Document(page_content=text) 
        for text in documents
    ]

    # Proceed with vectorstore and FAISS index creation
    vectorstore = FAISS.from_documents(document_objects, OpenAIEmbeddings())
    vectorstore.save_local(faiss_path)
    return vectorstore

def load_faiss_index(faiss_path):
    return FAISS.load_local(faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)

def add_to_vectorstore(texts, index_path=FAISS_PATH):
    """
    기존 벡터 스토어에 데이터를 추가하고 업데이트.

    Args:
        texts (list): 추가할 텍스트 청크 리스트.
        metadatas (list): 추가할 텍스트 청크에 해당하는 메타데이터 리스트.
        index_path (str): 기존 벡터 스토어 저장 경로.

    Returns:
        FAISS: 업데이트된 벡터 스토어 객체.
    """
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    vectorstore.add_texts(texts)
    vectorstore.save_local(index_path)


def search_vectorstore(
    vectorstore: VectorStore,
    query: str,
    k: int = TOP_K_RESULTS
) -> List[str]:
    """
    벡터 데이터베이스에서 유사한 문서를 검색하는 함수.
    
    Parameters:
        vectorstore (VectorStore): 로드된 벡터 데이터베이스 객체 (FAISS, Chroma 등).
        query (str): 검색 쿼리.
        k (int): 반환할 문서의 개수. 기본값은 5.
    
    Returns:
        List[str]: 검색된 문서 내용의 리스트.
    """
    try:
        # 검색 수행
        results = vectorstore.similarity_search(query, k=k)
        # 검색 결과에서 문서 내용 추출
        return [result.page_content for result in results]
    except Exception as e:
        print(f"검색 중 오류 발생: {e}")
        return []

