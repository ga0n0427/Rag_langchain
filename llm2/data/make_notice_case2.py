import openai
import pandas as pd
import random
import asyncio
import time

# OpenAI API 키 설정
openai.api_key = "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"


# 데이터 리스트
dong_list = [f"온양 {i}동" for i in range(1, 7)]  # 온양 1동~6동
store_list = [f"{i}호점" for i in range(1, 4)]  # 1~3호점
event_types = [
    "세일 이벤트",
    "경품 추첨",
    "체험 행사",
    "특별 프로모션",
    "무료 샘플 증정",
    "크리스마스 한정 이벤트",
    "신규 회원 이벤트"
]  # 7가지 이벤트 유형
dates = [
    f"2024-{month:02d}-{day:02d}" for month, day in zip(range(1, 10, 2), range(5, 50, 5))
]  # 날짜 리스트

# GPT API 요청 함수
async def generate_event(title, dong, store, event_type, date):
    prompt = f"""
    제목: {title}
    이벤트 위치: {dong} {store}
    이벤트 유형: {event_type}
    이벤트 날짜: {date}
    위 정보를 바탕으로, 자연스러운 이벤트 공지를 작성해주세요. 내용은 300자 이내로 작성되며, 고객이 쉽게 이해할 수 있도록 간결하게 작성해주세요.
    """
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for writing event notices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=300,
            temperature=0.7,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return "생성 실패"

# 이벤트 데이터 생성 및 저장
async def create_event_data(num_events=18, output_file="event_notices.csv"):
    data = []
    for i in range(num_events):
        dong = random.choice(dong_list)
        store = random.choice(store_list)
        event_type = random.choice(event_types)
        date = random.choice(dates)
        title = f"[이벤트] {event_type} - {dong} {store}"

        # API 요청
        content = await generate_event(title, dong, store, event_type, date)
        time.sleep(1)  # 1초 대기 (API 요청 제한 방지)

        # 데이터 저장
        data.append({
            "title": title,
            "content": content
        })
        print(data)
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"{output_file} 파일로 저장 완료!")

# 비동기 실행
if __name__ == "__main__":
    asyncio.run(create_event_data())
