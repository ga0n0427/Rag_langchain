import threading
from langchain.chains import ConversationChain
from langchain_openai import ChatOpenAI
from typing import List
from use_query import rewrite_query, generate_answer, check_hallucination, is_popup_store_question, search_and_cohere_rerank
from vectorstore_manage import load_faiss_index
from utils import preprocess_spacing_and_whitespace, format_docs
from config import DEFAULT_LLM_MODEL, TEMPERATURE, FAISS_PATH

# LangChain 초기화
llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=TEMPERATURE)

def generate_chat(user_message: str) -> str:
    """
    사용자 메시지를 받아서 RAG 워크플로를 통해 응답을 생성하는 함수.

    Args:
        user_message (str): 사용자 메시지.

    Returns:
        str: 최종 생성된 응답.
    """

    query = preprocess_spacing_and_whitespace(user_message)
    # 스레드 결과 저장용 변수
    rewritten_query = None
    is_popup_related = None

    def rewrite_query_thread():
        nonlocal rewritten_query
        rewritten_query = rewrite_query(query, llm)

    def classify_popup_thread():
        nonlocal is_popup_related
        is_popup_related = is_popup_store_question(query, llm)
    # 벡터스토어 로드
    vectorstore = load_faiss_index(FAISS_PATH)
    

    # 스레드 생성
    rewrite_thread = threading.Thread(target=rewrite_query_thread)
    classify_thread = threading.Thread(target=classify_popup_thread)

    # 스레드 시작
    rewrite_thread.start()
    classify_thread.start()

    # 스레드 완료 대기
    rewrite_thread.join()
    classify_thread.join()

    # 팝업스토어 관련 여부 판단
    if is_popup_related == "no":
        return "팝업스토어와 관련된 질문을 해주세요."

    # 팝업스토어 질문일 경우 RAG 워크플로 진행
    # Step 1: Cohere를 사용한 Rerank
    reranked_results = search_and_cohere_rerank(rewritten_query, vectorstore)
    # Step 2: 답변 생성
    generated_answer = generate_answer(reranked_results, rewritten_query, llm)

    # Step 3: 할루시네이션 검사
    hallucination_result = check_hallucination(reranked_results, generated_answer, llm)

    # Step 4: 응답 구성
    if hallucination_result == "no":
        generated_answer += "\n\n※ 이 답변은 정확하지 않을 수 있습니다."

    return generated_answer
