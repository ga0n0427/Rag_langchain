import os
import psycopg2
import select
from dotenv import load_dotenv
from .fetch_data import process_notice_data, process_qna_data

# .env 파일 로드
load_dotenv()

# PostgreSQL 연결 정보
DB_HOST = os.getenv("DB_HOST")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT")



def listen_for_notifications():
    """
    PostgreSQL 알림을 수신하고 채널별로 데이터 처리를 수행합니다.
    """
    try:
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()
        cursor.execute("LISTEN new_data_channel;")
        cursor.execute("LISTEN another_data_channel;")
        print("Listening for notifications...")

        while True:
            if select.select([connection], [], [], 5) == ([], [], []):
                pass
            else:
                connection.poll()
                while connection.notifies:
                    notify = connection.notifies.pop(0)
                    if notify.channel == "new_data_channel":
                        print("Processing notice table...")
                        process_notice_data(connection)
                    elif notify.channel == "another_data_channel":
                        print("Processing qna table...")
                        process_qna_data(connection)

    except psycopg2.Error as e:
        print(f"Database error: {e}")
    except KeyboardInterrupt:
        print("Stopped listening.")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()

