from use_query import cohere_rerank_only, generate_answer, generate_feedback, query_rewrite
from config import FAISS_PATH, TOP_K_RESULTS, llm, MyState
from dotenv import load_dotenv
from langchain_upstage import UpstageGroundednessCheck
from vectorstore_manage import load_faiss_index

load_dotenv()

def update_contexts(state: MyState) -> MyState:
    """
    context를 업데이트하는 노드 함수
    """
    vectorstore = load_faiss_index(FAISS_PATH)
    return update_contexts_rerank(
        state=state,
        vectorstore=vectorstore,
        top_n=TOP_K_RESULTS
    )

def update_contexts_rerank(
    state: MyState,
    vectorstore,
    top_n: int = TOP_K_RESULTS
) -> MyState:
    """
    기존 state의 context를 Cohere로 재정렬된 결과로 업데이트하는 함수.
    
    :param state: 업데이트할 MyState 객체.
    :param vectorstore: FAISS 벡터스토어 객체.
    :param top_n: 반환할 문서 개수 (기본값: 3).
    :return: context가 업데이트된 MyState 객체.
    """
    try:
        # state에서 query 가져오기
        query = state.get("question", "")
        # cohere_rerank_only를 사용하여 문서 재정렬
        reranked_results = cohere_rerank_only(query, vectorstore, top_n=top_n)
        # context 업데이트
        if reranked_results:
            # Document 객체 리스트를 문자열 리스트로 변환
            state["context"] = reranked_results

        else:
            print("No reranked results to update context.")
        return state
    except Exception as e:
        print(f"Error updating context: {e}")
        return state
    
def update_feedback(state: MyState):
    try:
        state = generate_feedback(state, llm)
        return state
    except Exception as e:
        print(f"Error during feedback {e}")
        return {"error": str(e)}
def update_query(state: MyState):
    try:
        state = query_rewrite(state, llm)
        return state
    except Exception as e:
        print(f"Error during update query {e}")
        return {"error": str(e)}
def check_groundedness(state: MyState) -> MyState:
    try:
        # UpstageGroundednessCheck 객체 초기화
        checker = UpstageGroundednessCheck()

        # 평가를 위한 입력 데이터 구성
        input_data = {
            'context': state.get('context', []),  # context가 없으면 빈 리스트
            'answer': state.get('question', "")  # question이 없으면 빈 문자열
        }
        # Groundedness 평가 수행
        result = checker.invoke(input_data)
        state['check_evaluated'] = result
        return state
    except Exception as e:
        print(f"Error during groundedness evaluation: {e}")
        return state
    
def update_answer(state: MyState) -> MyState:
    try:
        # state에서 query 가져오기
        query = state.get("question", "")
        context = state.get("context", "")

        # 새로운 답변 생성
        answer = generate_answer(context, query, llm)

        # 'answer'를 리스트로 관리
        if 'answer' not in state:
            state['answer'] = []  # 초기화

        # 중복되지 않도록 새로운 답변 추가
        if answer not in state['answer']:
            state['answer'].append(answer)

        return state
    except Exception as e:
        print(f"Error updating answer: {e}")
        return state
    
