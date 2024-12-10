from langgraph.graph import StateGraph, START, END
from t_langgraph import update_contexts, check_groundedness, update_answer, update_feedback, update_query
from config import MyState
def low_score_condition(state: MyState):
    query_evaluated = state.get('query_evaluated', 1.0)
    try:
        score = float(query_evaluated)
        return 'update_query' if score <= 0.7 else END
    except (ValueError, TypeError):
        return END

def graph_chat(query: str):
    # 초기 상태 설정
    initial_state = MyState(
        context=[],                 
        answer=[],                  
        question=query,             
        query_evaluated=None,       
        check_evaluated=None        
    )
    
    # StateGraph 인스턴스 생성
    graph = StateGraph(MyState)

    # 노드 추가
    graph.add_node("update_context", update_contexts)
    graph.add_node("update_feedback", update_feedback)
    graph.add_node("update_answer", update_answer)
    graph.add_node("update_query", update_query)
    
    # 기본 엣지 정의
    graph.add_edge(START, "update_context")
    graph.add_edge("update_context", "update_answer")
    graph.add_edge("update_answer", "update_feedback")

    # 조건부 엣지 수정
    graph.add_conditional_edges(
        "update_feedback", 
        low_score_condition
    )

    # 추가 엣지 정의
    graph.add_edge("update_query", "update_context")  # 쿼리 재작성 후 컨텍스트 업데이트
    graph.add_edge("update_feedback", END)  # 기본 경로는 END로 연결

    try:
        # 그래프 컴파일
        app = graph.compile()

        # 상태 실행
        final_state = app.invoke(initial_state)

        # 최종 답변 반환
        return final_state['answer'][-1] if final_state['answer'] else "No answer generated"

    except Exception as e:
        print(f"Graph execution error: {e}")
        return f"An error occurred: {e}"