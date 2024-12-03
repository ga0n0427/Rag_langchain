from dotenv import load_dotenv
import os
import psycopg2
from vectorstore_manage import add_to_vectorstore
from utils import process_documents, convert_to_documents, replace_t
from config import CHUNK_SIZE, CHUNK_OVERLAP, FAISS_PATH
from langchain.schema import Document

def fetch_notice_data(connection):
    """
    notice 테이블에서 데이터를 가져오고 is_checked 상태를 업데이트합니다.
    """
    try:
        cursor = connection.cursor()
        fetch_query = "SELECT title, detail FROM notice WHERE is_checked = FALSE;"
        cursor.execute(fetch_query)
        rows = cursor.fetchall()  # [(title, detail), ...]

        if rows:
            update_query = "UPDATE notice SET is_checked = TRUE WHERE title = %s;"
            for row in rows:
                cursor.execute(update_query, (row[0],))
            connection.commit()

        return rows  # [(title, detail), ...]

    except psycopg2.Error as e:
        print(f"Error fetching notice data: {e}")
        connection.rollback()
        return []
    finally:
        if cursor:
            cursor.close()


def fetch_qna_data(connection):
    """
    qna 테이블에서 데이터를 가져오고 is_checked 상태를 업데이트합니다.
    반환값은 (question, question + answer) 형식입니다.
    """
    try:
        cursor = connection.cursor()
        fetch_query = "SELECT question, answer, email, date FROM qna WHERE is_checked = FALSE;"
        cursor.execute(fetch_query)
        rows = cursor.fetchall()  # [(question, answer, email, date), ...]

        if rows:
            update_query = "UPDATE qna SET is_checked = TRUE WHERE email = %s;"
            for row in rows:
                cursor.execute(update_query, (row[2],))  # row[2]는 email
            connection.commit()
            print(f"{len(rows)}개의 행이 qna 테이블에서 업데이트되었습니다.")

        # question과 question + answer 값을 생성
        combined_rows = [
            (row[0], f"{row[0]}: {row[1]}")  # (question, question + answer)
            for row in rows
        ]

        return combined_rows  # [(question, question + answer), ...]

    except psycopg2.Error as e:
        print(f"Error fetching qna data: {e}")
        connection.rollback()
        return []
    finally:
        if cursor:
            cursor.close()

def process_notice_data(connection):
    """
    notice 테이블의 데이터를 처리하고 벡터 데이터베이스에 저장합니다.
    """
    # Step 1: 데이터를 가져옴
    data = fetch_notice_data(connection)

    if data:
        # Step 2: Document 객체로 변환
        documents = convert_to_documents(data)

        # Step 3: 텍스트 청크 생성
        chunks = process_documents(
            file_content=documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_semantic=True,
        )

        # Step 4: 텍스트로 변환 (Document 객체가 아닌 경우 처리)
        texts = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                texts.append(chunk.page_content)  # Document 객체에서 page_content 추출
            else:
                texts.append(chunk)  # 이미 텍스트인 경우 그대로 추가
        texts = replace_t(texts)  # 탭 제거
        # Step 5: 벡터 스토어에 텍스트 추가
        add_to_vectorstore(texts, index_path=FAISS_PATH)



def process_qna_data(connection):
    """
    qna 테이블의 데이터를 처리하고 벡터 데이터베이스에 저장합니다.
    """
    # Step 1: 데이터를 가져옴
    data = fetch_qna_data(connection)

    if data:
        # Step 2: Document 객체로 변환
        documents = convert_to_documents(data)

        # Step 3: 텍스트 청크 생성
        chunks = process_documents(
            file_content=documents,
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            use_semantic=True,
        )

        texts = []
        for chunk in chunks:
            if isinstance(chunk, Document):
                texts.append(chunk.page_content)  # Document 객체에서 page_content 추출
            else:
                texts.append(chunk)  # 이미 텍스트인 경우 그대로 추가
        
        texts = replace_t(texts)  # 탭 제거
        # Step 5: 벡터 스토어에 텍스트 추가
        add_to_vectorstore(texts, index_path=FAISS_PATH)
