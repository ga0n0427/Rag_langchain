import random
import openai
import pandas as pd 
import time
import asyncio
# OpenAI API 키 설정
openai.api_key = "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"

# 데이터 조합 리스트
dong_list = [f"온양 {i}동" for i in range(1, 7)]  # 온양 1동~6동
store_list = [f"{i}호점" for i in range(1, 7)]  # 1~6호점
recruitment_periods = [
    f"2024-{month:02d}-{day:02d} ~ 2024-{month+1:02d}-{day+7:02d}"
    for month, day in zip(range(1, 10, 2), range(1, 10))
] * 2  # 18개 모집 기간, 2번 반복
application_methods = [f"https://apply.popupstore-{i}.com" for i in range(1, 21)]  # 20개 신청 방법
extra_info_samples = [
    "운영자는 공간 임대료 할인 혜택을 받을 수 있습니다.",
    "참여자에게 무료 인테리어 지원이 포함됩니다.",
    "선정된 운영자는 홍보 지원을 받을 수 있습니다.",
    "운영 기간 동안 판매 매출 분석 서비스를 제공합니다."
]

# GPT API 요청 함수 (비동기)
async def generate_notice(title, dong, store, period, method, extra_info):
    prompt = f"""
    제목: {title}
    위치: {dong} {store}
    모집 기간: {period}
    신청 방법: {method}
    추가 정보: {extra_info}
    위 정보를 활용하여 팝업스토어 신청 공지를 작성해주세요. 내용은 자연스럽고 간결하게 500자 이내로 작성해주세요.
    """
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for writing notices."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.5,
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error: {e}")
        return "생성 실패"

# 데이터 생성 및 저장
async def create_popup_data(num_samples=36, output_file="popup_store_notices.csv"):
    data = []
    for i in range(num_samples):
        title = f"팝업스토어 모집: {random.choice(dong_list)} {random.choice(store_list)}"
        dong = random.choice(dong_list)
        store = random.choice(store_list)
        period = recruitment_periods[i % len(recruitment_periods)]
        method = random.choice(application_methods)
        extra_info = random.choice(extra_info_samples)

        # API 요청
        notice_content = await generate_notice(title, dong, store, period, method, extra_info)
        time.sleep(1)  # 1초 대기 (API 요청 제한 방지)

        # 데이터 저장
        data.append({
            "title": title,
            "content": notice_content
        })
        print(data)
    # 데이터프레임 생성
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")
    print(f"{output_file} 파일로 저장 완료!")

# 비동기 실행
if __name__ == "__main__":
    asyncio.run(create_popup_data())