from dotenv import load_dotenv
import os
import psycopg2
from vectorstore_manage import add_to_vectorstore
from utils import process_documents, convert_to_documents
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
    except Exception as e:
        print(f"Unexpected error in fetch_notice_data: {e}")
        return []
    finally:
        if cursor:
            cursor.close()


def process_notice_data(connection):
    """
    notice 테이블의 데이터를 처리하고 벡터 데이터베이스에 저장합니다.
    """
    try:
        # Step 1: 데이터를 가져옴
        data = fetch_notice_data(connection)
        if not data:
            return

        # Step 2: Document 객체로 변환
        try:
            documents = convert_to_documents(data)
        except Exception as e:
            print(f"Error converting data to Document objects: {e}")
            return

        # Step 3: 텍스트 청크 생성
        try:
            chunks = process_documents(
                file_content=documents,
                chunk_size=CHUNK_SIZE,
                chunk_overlap=CHUNK_OVERLAP,
                use_semantic=False,
            )
        except Exception as e:
            print(f"Error processing documents: {e}")
            return

        # Step 4: 벡터 스토어에 텍스트 추가
        try:
            add_to_vectorstore(chunks, index_path=FAISS_PATH)
        except Exception as e:
            print(f"Error adding chunks to vector store: {e}")

    except Exception as e:
        print(f"Unexpected error in process_notice_data: {e}")
