from utils.loader import load_documents_from_folder
from vectorstore_manage.manager import load_faiss_index, build_faiss_index
from use_query.reranker import search_and_rerank
from use_query.query_handler import generate_response
from utils.chunker import process_documents
from deepeval.metrics import GEval, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval import evaluate

def test_load_documents(txt_path):
    """문서 로드 테스트"""
    documents = load_documents_from_folder(txt_path)
    assert len(documents) > 0, "No documents loaded from the folder"
    return documents

def test_faiss_index(txt_path, faiss_path):
    """FAISS 인덱스 빌드 및 로드 테스트"""
    try:
        vectorstore = load_faiss_index(faiss_path)
    except:
        documents = test_load_documents(txt_path)
        chunks = process_documents(file_content=documents, chunk_size=300, chunk_overlap=0, use_semantic=True)
        vectorstore = build_faiss_index(chunks, faiss_path)
    return vectorstore

def test_search_and_rerank(query, vectorstore, top_n=2):
    """BM25 검색 및 재정렬 테스트"""
    search_results = search_and_rerank(query, vectorstore, top_n)
    assert isinstance(search_results, list), "Expected search results to be a list"
    return search_results

def test_generate_response(search_results, query):
    """응답 생성 테스트"""
    response = generate_response(search_results, query)
    assert isinstance(response, str), "Expected response to be a string"
    return response

def test_evaluation(gt_answer, pred_answer, search_results):
    """평가 메트릭 테스트"""
    test_case_correctness = LLMTestCase(
        input="What is the capital of Spain?",
        expected_output=gt_answer,
        actual_output=pred_answer,
    )

    correctness_metric = GEval(
        name="Correctness",
        model="gpt-4",
        evaluation_params=[LLMTestCaseParams.EXPECTED_OUTPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        evaluation_steps=["Determine whether the actual output is factually correct based on the expected output."]
    )
    correctness_metric.measure(test_case_correctness)
    #답변이 질문에 충실한지 
    faithfulness_metric = FaithfulnessMetric(
        threshold=0.7,
        model="gpt-4",
        include_reason=False #평가 이유 
    )
    faithfulness_test_case = LLMTestCase(
        input="What is 3+3?",
        actual_output=pred_answer,
        retrieval_context=search_results
    )
    faithfulness_metric.measure(faithfulness_test_case)
    #Contextual Relevancy 얼마나 질문과 관련있는지 
    contextual_relevancy_metric = ContextualRelevancyMetric(
        threshold=1,
        model="gpt-4",
        include_reason=True
    )
    relevance_test_case = LLMTestCase(
        input="What if these shoes don't fit?",
        actual_output=pred_answer,
        retrieval_context=search_results
    )
    contextual_relevancy_metric.measure(relevance_test_case)

    # Running multiple tests together
    evaluate(
        test_cases=[test_case_correctness, relevance_test_case],
        metrics=[correctness_metric, faithfulness_metric, contextual_relevancy_metric]
    )
    return correctness_metric.score, faithfulness_metric.score, contextual_relevancy_metric.score
