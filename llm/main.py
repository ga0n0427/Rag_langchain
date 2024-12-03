from utils import load_documents_from_folder, process_documents, preprocess_spacing_and_whitespace
from vectorstore_manage import load_faiss_index, build_faiss_index
from use_query import search_with_faiss, search_and_rerank, search_and_cohere_rerank, generate_response
from config import FAISS_PATH, CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_RESULTS, DEFAULT_LLM_MODEL, TEMPERATURE
from langchain_openai import ChatOpenAI
def main():
    txt_path = "data"
    faiss_path = FAISS_PATH
    llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=TEMPERATURE)
    
    # Step 3: Build or Load FAISS Index
    
    try:
        vectorstore = load_faiss_index(faiss_path)
    except:
        # Step 1: Load data
        documents = load_documents_from_folder(txt_path)

        # Step 2: Create chunks with token + semantic
        chunks = process_documents(file_content = documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, use_semantic=True)

        vectorstore = build_faiss_index(chunks, faiss_path)

    # Step 4: Process query
    query = input("Enter your question: ")
    query = preprocess_spacing_and_whitespace(query)
    search_results = search_and_cohere_rerank(query, vectorstore, TOP_K_RESULTS)
    if search_results == None:
       print("정보가 정확하지 않을 수 있습니다.") 
    else:
        print(search_results)
        #if
        # Step 5: Generate response
        check, response, check_2 = generate_response(search_results, query, llm)
        print(f"Answer: {response}")
        print(f"Check:{check_2}")
        print(f"Check:{check}")
    """
     # Step 4: Evaluation
    gt_answer = "Madrid is the capital of Spain."  # Example ground truth
    pred_answer = response  # Predicted answer from the model
    correctness_score, faithfulness_score, contextual_score = test_evaluation(gt_answer, pred_answer, search_results)
    
    print(f"Correctness Score: {correctness_score}")
    print(f"Faithfulness Score: {faithfulness_score}")
    print(f"Contextual Relevancy Score: {contextual_score}")
    """
if __name__ == "__main__":
    main()

