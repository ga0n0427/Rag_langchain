from typing import List
from langchain.schema import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from config import CHUNK_SIZE, CHUNK_OVERLAP

def semantic_chunking(chunks: List[str], 
                      breakpoint_threshold_type: str = "percentile", 
                      breakpoint_threshold_amount: int = 70) -> List[str]:
    """
    의미 기반으로 텍스트를 분할합니다.
    :param chunks: 토큰 기반으로 분할된 텍스트 리스트
    """
    # SemanticChunker 초기화
    semantic_chunker = SemanticChunker(
        embeddings=OpenAIEmbeddings(),
        breakpoint_threshold_type=breakpoint_threshold_type,
        breakpoint_threshold_amount=breakpoint_threshold_amount
    )

    # 의미 기반 분할 수행
    semantic_chunks = []
    for chunk in chunks:
        semantic_chunks.extend(semantic_chunker.split_text(chunk))
    return semantic_chunks


def token_split(documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    토큰 기반으로 텍스트를 분할합니다.
    :param documents: Document 객체 리스트
    :param chunk_size: 청크 크기
    :param chunk_overlap: 청크 간 중복
    :return: 토큰 기반으로 분할된 텍스트 리스트
    """
    token_splitter = CharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    token_chunks = []
    for doc in documents:
        token_chunks.extend(token_splitter.split_text(doc.page_content))
    return token_chunks


def create_chunks(documents: List[Document], 
                  chunk_size: int = CHUNK_SIZE, 
                  chunk_overlap: int = CHUNK_OVERLAP, 
                  use_semantic: bool = True,
                  breakpoint_threshold_type: str = "percentile", 
                  breakpoint_threshold_amount: int = 70) -> List[Document]:
    """
    혼합된 청크 생성: 토큰 기반과 의미 기반.
    :param documents: Document 객체 리스트
    :return: 청크 단위로 분할된 Document 리스트
    """
    token_chunks = token_split(documents, chunk_size, chunk_overlap)

    if use_semantic:
        final_chunks = semantic_chunking(
            token_chunks,
            breakpoint_threshold_type=breakpoint_threshold_type,
            breakpoint_threshold_amount=breakpoint_threshold_amount
        )
    else:
        final_chunks = token_chunks

    # 최종 청크를 Document로 변환하며 원래 파일의 metadata 유지
    chunked_documents = [Document(page_content=chunk)  for doc in documents for chunk in final_chunks]
    return chunked_documents


def process_documents(file_content: List[Document], 
                      chunk_size: int = CHUNK_SIZE, 
                      chunk_overlap: int = CHUNK_OVERLAP, 
                      use_semantic: bool = True) -> List[Document]:
    """
    파일 내용을 처리하여 청크를 생성합니다.
    :param file_content: 텍스트 파일 내용 (Document 리스트)
    :param chunk_size: 청크 크기 (토큰 수 기준, 기본값: 300)
    :param chunk_overlap: 청크 간 중복 토큰 수 (기본값: 0)
    :param use_semantic: 의미 기반 청크 사용 여부 (기본값: True)
    :return: 처리된 Document 리스트
    """
    # 유효성 검사
    if not file_content or not isinstance(file_content, list) or not all(isinstance(doc, Document) for doc in file_content):
        raise ValueError("file_content는 Document 객체의 비어 있지 않은 리스트여야 합니다.")

    # 청크 생성
    chunks = create_chunks(file_content, chunk_size, chunk_overlap, use_semantic)

    # 중복 제거
    unique_chunks = []
    seen_content = set()
    for chunk in chunks:
        if chunk.page_content not in seen_content:
            seen_content.add(chunk.page_content)
            unique_chunks.append(chunk)

    return unique_chunks

