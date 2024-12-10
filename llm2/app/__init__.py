import threading
from flask import Flask
from dotenv import load_dotenv
import os
from data_manager import listen_for_notifications
from vectorstore_manage import check_vectorstore
from config import FAISS_PATH

# .env 파일 로드
load_dotenv()

def create_app():
    """Flask 앱 생성 및 초기화"""
    app = Flask(__name__)

    # 환경 변수 설정
    app.config['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')
    # 백그라운드 스레드에서 listen_for_notifications 실행
    notification_thread = threading.Thread(target=listen_for_notifications, daemon=True)
    notification_thread.start()

    # 벡터스토어 확인
    check_vectorstore(FAISS_PATH)

    # 라우트 설정
    from app.routes import setup_routes
    setup_routes(app)

    return app
