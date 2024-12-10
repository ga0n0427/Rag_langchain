import requests

# Flask 서버의 URL 및 엔드포인트
url = ""  # Flask 서버 주소와 엔드포인트

# POST 요청에 보낼 데이터
data = {
    "id": "12345",
    "message": "온양 6동 1호점에서 팝업스토어 모집은 언제까지 해"

}


# POST 요청 보내기
response = requests.post(url, json=data)

# 응답 결과 출력
if response.status_code == 200:
    print("Response from Flask server:", response.json())
else:
    print(f"Error: {response.status_code}, {response.text}")
