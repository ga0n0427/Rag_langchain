from .prompts import (
    query_rewrite_template, 
    answer_generation_template, 
    hallucination_prompt_template, 
    popup_store_verification_template, 
    popup_store_classification_template,
    feedback_prompt_template,
    query_rewrite_prompt_template
)
import json
from config import MyState

def query_rewrite(state: MyState, llm) -> MyState:
    """
    검색 결과와 피드백을 기반으로 쿼리를 재작성합니다.
    
    Args:
        state (MyState): 상태 객체
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        MyState: 재작성된 쿼리가 포함된 업데이트된 상태 객체
    """
    try:
        # State에서 데이터 가져오기
        original_query = state.get('question', '')
        query_feedback = state.get('query_feedback', '')
        print(original_query)
        # 템플릿 실행 및 LLM 호출
        query_rewriter = query_rewrite_prompt_template | llm
        rewritten_query_response = query_rewriter.invoke({
            "original_query": original_query,
            "feedback": query_feedback
        })

        # 재작성된 쿼리 추출
        rewritten_query = rewritten_query_response.content.strip()
        # 상태 업데이트
        state['question'] = rewritten_query
        return state
    
    except Exception as e:
        print(f"Error rewriting query: {e}")
        state['rewritten_query'] = None
        return state

    
def generate_answer(search_results: str, rewritten_query: str, llm) -> str:
    """
    검색 결과와 재작성된 쿼리를 기반으로 답변을 생성합니다.
    
    Args:
        search_results (str): 검색된 문서 문자열
        rewritten_query (str): 재작성된 사용자 질문
        llm (ChatOpenAI): LangChain LLM 객체
    
    Returns:
        str: 생성된 답변
    """
    try:
        # 답변 생성기 호출
        answer_generator = answer_generation_template | llm
        generated_answer = answer_generator.invoke({
            "context": search_results,
            "rewritten_query": rewritten_query
        }).content
        
        return generated_answer
    
    except Exception as e:
        raise ValueError(f"답변 생성 중 오류 발생: {e}")

def generate_feedback(state: MyState, llm) -> dict:
    """
    생성된 답변을 컨텍스트와 질문에 대해 평가하고, 피드백으로 상태를 업데이트합니다.

    매개변수(Args):
    state (dict): 'context', 'question', 'answer'를 포함하는 상태 객체.
    llm (ChatOpenAI): 피드백을 생성하기 위해 사용하는 LangChain LLM 객체.
    feedback_prompt_template (str): 피드백 프롬프트 템플릿.
    반환값(Returns):
    dict: 'reliability_score'와 'feedback'이 추가된 업데이트된 상태 객체.
    """
    try:
        # State에서 입력 데이터 가져오기
        context = state.get('context', '')
        question = state.get('question', '')
        answer = state.get('answer', '')

        # 피드백 프롬프트 작성
        feedback_prompt = feedback_prompt_template.format(
            context=context,
            question=question,
            answer=answer
        )

        # LangChain LLM 호출
        feedback_response = llm.invoke(feedback_prompt)  # 문자열로 전달
        
        # LLM의 응답을 파싱하여 신뢰 점수와 피드백 추출
        feedback = eval(feedback_response.content)  # LLM이 반환한 문자열을 리스트로 변환

        # 결과 검증
        if (
            isinstance(feedback, list) and 
            len(feedback) == 2 and 
            isinstance(feedback[0], float) and 
            0 < feedback[0] < 1 and 
            isinstance(feedback[1], str)
        ):
            # State 업데이트
            state['query_evaluated'] = feedback[0]
            state['query_feedback'] = feedback[1]
            return state
        else:
            raise ValueError("Invalid feedback format received from LLM.")
    except Exception as e:
        print(f"Error generating feedback: {e}")
        state['query_evaluated'] = None
        state['query_feedback'] = f"Error generating feedback: {e}"
        return state












def rewrite_query(original_query: str, llm) -> str:
    """
    사용자 입력 쿼리를 재작성합니다.

    Args:
        original_query (str): 원본 사용자 쿼리
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        str: 재작성된 쿼리
    """
    try:
        query_rewriter = query_rewrite_template | llm
        rewritten_query = query_rewriter.invoke({"original_query": original_query}).content
        return rewritten_query
    except Exception as e:
        raise ValueError(f"Error rewriting query: {e}")

def check_hallucination(search_results: list, generated_answer: str, llm) -> str:
    """
    생성된 답변이 제공된 문서에서 정확히 유래했는지 평가합니다.

    Args:
        search_results (list): 검색된 문서 리스트
        generated_answer (str): 생성된 답변
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        str: 할루시네이션 평가 결과 ("yes" 또는 "no")
    """
    try:
        context = json.dumps([{"text": doc.page_content} for doc in search_results])
        hallucination_checker = hallucination_prompt_template | llm
        hallucination_response = hallucination_checker.invoke({
            "documents": context,
            "generation": generated_answer
        }).content.strip().lower()

        if hallucination_response not in ["yes", "no"]:
            raise ValueError(f"Unexpected response from hallucination checker: {hallucination_response}")

        return hallucination_response
    except Exception as e:
        raise ValueError(f"Error checking hallucination: {e}")

def is_query_in_reranked_results(search_results: list, query: str, llm) -> str:
    """
    리랭크된 검색 결과와 쿼리를 기반으로 팝업스토어와 관련된 질문인지 판단합니다.

    Args:
        search_results (list): 리랭크된 문서 리스트
        query (str): 사용자 입력 질문
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        str: "yes" 또는 "no" (결과 포함 여부)
    """
    try:
        # 검색 결과에서 문서 내용 추출
        context = json.dumps([{"text": doc.page_content} for doc in search_results])

        # LLM과 프롬프트 결합
        verifier = popup_store_verification_template | llm

        # LLM 호출
        verification_response = verifier.invoke({"context": context, "query": query}).content.strip().lower()

        # 결과 검증
        if verification_response not in ["yes", "no"]:
            raise ValueError(f"Unexpected response from verifier: {verification_response}")

        return verification_response
    except Exception as e:
        raise ValueError(f"Error verifying query in results: {e}")
def generate_response(search_results, query, llm):
    """
    검색 결과와 쿼리를 기반으로 답변을 생성하고 할루시네이션을 체크합니다.

    Args:
        search_results (list): 검색된 문서 리스트
        query (str): 사용자 입력 질문
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        tuple: (is_hallucination: bool, generated_answer: str, hallucination_response: str)
    """
    try:
        # Step 1: 쿼리 재작성
        rewritten_query = rewrite_query(query, llm)

        # Step 2: 답변 생성
        generated_answer = generate_answer(search_results, rewritten_query, llm)

        # Step 3: 할루시네이션 체크
        hallucination_response = check_hallucination(search_results, generated_answer, llm)

        # 할루시네이션 결과를 Boolean 값으로 변환
        is_hallucination = hallucination_response == "no"

        return is_hallucination, generated_answer, hallucination_response

    except Exception as e:
        raise ValueError(f"Error in generate_response: {e}")

def is_popup_store_question(query: str, llm) -> str:
    """
    입력된 질문이 팝업스토어와 관련된 질문인지 판단합니다.

    Args:
        query (str): 사용자 입력 질문
        llm (ChatOpenAI): LangChain LLM 객체

    Returns:
        str: "yes" 또는 "no" (팝업스토어 관련 여부)
    """
    try:
        # 팝업스토어 판단 프롬프트와 LLM 결합
        classifier = popup_store_classification_template | llm

        # LLM을 통해 판단 수행
        classification_response = classifier.invoke({"query": query}).content.strip().lower()

        # 결과 검증
        if classification_response not in ["yes", "no"]:
            raise ValueError(f"Unexpected response from classifier: {classification_response}")

        return classification_response
    except Exception as e:
        raise ValueError(f"Error classifying query: {e}")