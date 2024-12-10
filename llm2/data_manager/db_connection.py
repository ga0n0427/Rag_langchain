import os
import psycopg2
import select
from dotenv import load_dotenv
from .fetch_data import process_notice_data

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
    connection = None
    cursor = None
    try:
        # 데이터베이스 연결
        connection = psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )
        connection.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = connection.cursor()

        # 알림 채널 등록
        try:
            cursor.execute("LISTEN new_data_channel;")
        except Exception as e:
            print(f"Error registering notification channel: {e}")
            return

        # 알림 수신 루프
        while True:
            try:
                if select.select([connection], [], [], 5) == ([], [], []):
                    continue  # 타임아웃
                connection.poll()
                while connection.notifies:
                    notify = connection.notifies.pop(0)
                    if notify.channel == "new_data_channel":
                        try:
                            process_notice_data(connection)
                        except Exception as e:
                            print(f"Error processing notification: {e}")
            except Exception as e:
                print(f"Error while polling notifications: {e}")
                break  # 심각한 오류 발생 시 루프 종료
    except psycopg2.Error as db_error:
        print(f"Database connection error: {db_error}")
    except KeyboardInterrupt:
        print("Stopped listening due to keyboard interrupt.")
    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
        print("Database connection closed.")

