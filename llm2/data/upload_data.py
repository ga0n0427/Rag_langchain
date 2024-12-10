import psycopg2
import pandas as pd
from datetime import datetime, timedelta

# 데이터베이스 연결 정보 설정
db_config = {
    'dbname': '',
    'user': '',
    'password': '',
    'host': '',
    'port': ''
}

# CSV 파일 경로
csv_file = 'combined_notices.csv'

# 데이터 읽기
combined_notices = pd.read_csv(csv_file)

# 연결 및 데이터 삽입
try:
    # PostgreSQL 연결
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # 초기값 설정
    start_id = 2  # ID 시작값
    start_time = datetime(2024, 12, 3, 11, 30, 0)  # 시작 시간
    time_increment = timedelta(seconds=10)  # 시간 증가 간격

    # 전체 데이터 삽입
    for i, row in combined_notices.iterrows():
        notice_id = start_id + i
        title = row['title']
        detail = row['detail']
        notice_date = start_time + i * time_increment
        is_checked = False

        # SQL 쿼리 작성 및 실행
        insert_query = """
        INSERT INTO notice (id, title, detail, date, is_checked)
        VALUES (%s, %s, %s, %s, %s)
        """
        cursor.execute(insert_query, (notice_id, title, detail, notice_date, is_checked))

    # 변경사항 커밋
    conn.commit()
    print("모든 데이터 삽입이 완료되었습니다.")

except Exception as e:
    print(f"오류 발생: {e}")

finally:
    # 연결 닫기
    if cursor:
        cursor.close()
    if conn:
        conn.close()
