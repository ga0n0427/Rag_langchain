from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from utils import replace_t_with_space
from typing import List
from langchain.vectorstores import VectorStore
from langchain.schema import Document
from config import FAISS_PATH, TOP_K_RESULTS
import faiss
import os
import os

def check_vectorstore(FAISS_PATH="embedding/faiss_index", metadatas=None):
    """
    벡터스토어가 없으면 새로 생성하고, 이미 존재하면 로드합니다.
    """
    try:
        # 디렉토리 경로 확인 및 생성
        if not os.path.exists(FAISS_PATH):
            os.makedirs(FAISS_PATH)
            print(f"디렉토리가 존재하지 않아 생성했습니다: {FAISS_PATH}")

        # 벡터스토어 로드
        if os.path.exists(os.path.join(FAISS_PATH, "index.faiss")):  # FAISS 저장 구조 확인
            print(f"기존 벡터스토어 로드 중: {FAISS_PATH}")
            return load_faiss_index(FAISS_PATH)  # 기존 벡터스토어 로드
        else:
            # 벡터스토어가 없을 경우 새로 생성
            print(f"벡터스토어가 존재하지 않음. 새로 생성합니다: {FAISS_PATH}")
            return create_vectorstore_text(["g"], metadatas, index_path=FAISS_PATH)

    except Exception as e:
        print(f"벡터스토어 생성 또는 로드 중 오류 발생: {e}")


def create_vectorstore_text(texts, metadatas=None, index_path=FAISS_PATH):
    """
    벡터 스토어를 생성하고 로컬에 저장.
    """
    try:
        if not texts:
            raise ValueError("입력 텍스트가 비어 있습니다.")
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_texts(texts, embeddings, metadatas=metadatas)

        # 벡터스토어를 로컬에 저장
        vectorstore.save_local(index_path)
        return vectorstore
    except Exception as e:
        print(f"Error creating vectorstore: {e}")


def build_faiss_index(documents, faiss_path: str):
    """
    주어진 문서들로 FAISS 인덱스를 생성합니다.
    """
    try:
        if not documents:
            raise ValueError("문서 리스트가 비어 있습니다.")
        document_objects = [
            Document(page_content=text.page_content, metadata=text.metadata)
            if isinstance(text, Document)
            else Document(page_content=text)
            for text in documents
        ]
        vectorstore = FAISS.from_documents(document_objects, OpenAIEmbeddings())
        vectorstore.save_local(faiss_path)
        return vectorstore
    except Exception as e:  
        print(f"Error building FAISS index: {e}")


def load_faiss_index(faiss_path):
    try:
        return FAISS.load_local(faiss_path, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    except Exception as e:
        print(f"Error loading FAISS index: {e}")
        return None


def add_to_vectorstore(documents, index_path=FAISS_PATH):
    """
    기존 벡터 스토어에 Document 객체를 추가하고 업데이트.
    """
    try:
        print(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        vectorstore.add_texts(texts, metadatas=metadatas)
        vectorstore.save_local(index_path)
        print(1)
    except Exception as e:
        print(f"Error updating vectorstore: {e}")
