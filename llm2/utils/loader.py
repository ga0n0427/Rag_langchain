from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import TextLoader
from pdfminer.high_level import extract_text
from .use_functions import replace_t_with_space
from .chunker import process_documents
from langchain.schema import Document
from typing import List
import os
from config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_PATH

def convert_to_documents(data):
    """
    튜플 리스트 데이터를 Document 객체 리스트로 변환합니다.
    :param data: [(title, detail), ...] 형식의 데이터
    :return: Document 객체 리스트
    """
    documents = []
    for title, detail in data:
        # \t를 공백으로 치환
        detail = detail.replace("\t", " ")

        if detail.strip():  # 빈 값 제외
            document = Document(
                page_content=detail,  # detail을 내용으로 사용
                metadata={"title": title}  # title을 메타데이터로 추가
            )
            documents.append(document)
    return documents

def load_documents_from_folder(folder_path: str) -> List[Document]:
    """
    폴더 내 모든 텍스트 파일을 로드하여 Document 객체 리스트를 반환하며,
    텍스트에서 탭(\t)을 공백(" ")으로 대체합니다.
    :param folder_path: 텍스트 파일들이 저장된 폴더 경로
    :return: 로드된 Document 객체 리스트
    """
    documents = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".txt"):  # 텍스트 파일만 처리
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
                    
                    # \t를 공백으로 치환
                    file_content = file_content.replace("\t", " ")
                    
                    if file_content.strip():  # 빈 파일 제외
                        documents.append(Document(page_content=file_content, metadata={"file_name": file}))
    
    return documents

def load_pdfs_from_folder(folder_path: str) -> List[Document]:
    """
    폴더 내 모든 PDF 파일을 읽어 Document 객체 리스트를 반환합니다.
    
    :param folder_path: PDF 파일들이 저장된 폴더 경로
    :return: 로드된 Document 객체 리스트
    """
    documents = []
    
    # 폴더 내 파일 탐색
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(".pdf"):  # PDF 파일만 처리
                file_path = os.path.join(root, file)
                try:
                    # PDF 파일에서 텍스트 추출
                    file_content = extract_text(file_path)
                    
                    # 빈 텍스트 제외
                    if file_content.strip():
                        documents.append(Document(page_content=file_content, metadata={"file_name": file}))
                except Exception as e:
                    print(f"Error reading {file}: {e}")
    
    return documents