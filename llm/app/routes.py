from flask import request, jsonify
from app.services.chat_service import generate_chat
import traceback
import json

def setup_routes(app):
    @app.route('/chat', methods=['POST'])
    def chat():
        try:
            # Content-Type 확인
            if request.content_type != "application/json":
                return jsonify({"error": "Content-Type must be application/json"}), 400

            # JSON 데이터 수동 파싱
            data = request.get_json(silent=True)
            if not data:
                try:
                    # 수동으로 JSON 파싱
                    data = json.loads(request.data.decode("utf-8"))
                except json.JSONDecodeError as e:
                    return jsonify({"error": "Invalid JSON format"}), 400

            # 필수 필드 검증
            if 'id' not in data or 'message' not in data:
                return jsonify({"error": "Missing 'id' or 'message' in request"}), 400

            user_id = data['id']
            user_message = data['message']
            print("Received Data -> ID:", user_id, "Message:", user_message)

            # Chat 응답 생성
            response = f"{user_message}"
            return jsonify({"id": user_id, "response": response})

        except Exception as e:
            print("Exception occurred:", str(e))
            return jsonify({"error": str(e)}), 500