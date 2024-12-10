from langchain.prompts import PromptTemplate

feedback_prompt_template = PromptTemplate(
    input_variables=["context", "answer", "question"],
    template="""
You are an evaluation assistant. Your task is to evaluate how well a generated answer matches the given context and question. Use the following criteria to objectively assess the answer:

1. **Keyword Match**: Check how many key terms or phrases in the context are accurately reflected in the answer.
2. **Coverage of Main Ideas**: Evaluate if the generated answer includes the core information or ideas present in the context.
3. **Accuracy**: Assess whether the answer correctly interprets or reflects the context and question without introducing irrelevant or incorrect information.

Provide:
1. A reliability score strictly between 0 and 1 (e.g., 0.75) based on the above criteria:
   - Weight each criterion equally to calculate the score.
2. Feedback including:
   - Specific areas where the answer could be improved or clarified.
   - Suggestions for rewriting the query to obtain a better answer.
   - Suggestions for modifying the question to make it more clear, specific, or aligned with the context.

### Input:
- Context: {context}
- Question: {question}
- Generated Answer: {answer}

### Output format:
[ Score between 0 and 1(float type), "Feedback including query rewriting suggestions and question modification suggestions(string type)"]

Please ensure the reliability score is strictly between 0 and 1 (e.g., 0.85). Feedback should include specific reasons for the score, actionable query rewriting suggestions, and recommendations for improving the question..
"""
)

query_rewrite_prompt_template =  PromptTemplate(
    input_variables=["original_query", "feedback"],
    template="""
You are an expert query rewriter. Your task is to rewrite the given query to improve its clarity, specificity, and alignment with the provided feedback. 

### Input:
- Original Query: {original_query}
- Feedback: {feedback}

### Output:
Return the rewritten query as a single string that incorporates the feedback and improves the original query.
"""
)
#당신은 팝업스토어 대여와 관련된 공지 사항을 바탕으로 질문에 정확하고 신뢰할 수 있는 답변을 제공하는 AI 어시스턴트입니다.
answer_generation_template = PromptTemplate(
    input_variables=["context", "rewritten_query"],
    template="""
    당신은 팝업스토어 대여와 관련된 공지 사항을 바탕으로 질문에 정확하고 신뢰할 수 있는 답변을 제공하는 AI 어시스턴트입니다.
    사용자가 제출한 질문과 context 내용을 바탕으로 가능한 한 정확하고 간결하게 답변을 작성하세요.
    만약 질문에 대한 정보가 context에 명확히 없다면, 이를 알리고 추가적인 정보를 제공하지 마세요.
    질문을 통해 모집, 이벤트, 정보 탐색등 분류를 잘 구별하여 답변하세요.
    그러나 context에 있는 정보를 최대한 활용하여 답변하세요.

    context:
    {context}

    질문:
    {rewritten_query}

    답변:"""
)






query_rewrite_template = PromptTemplate(
    input_variables=["original_query"],
    template="""
    - 간단하고 명확한 질문(두 문장 이하)은 재작성하지 말고 그대로 반환하세요.
    - 간단하고 명확한 질문(두 문장 이하)은 재작성하지 말고 그대로 반환하세요.
    - 간단하고 명확한 질문(두 문장 이하)은 재작성하지 말고 그대로 반환하세요.
    질문이 두 문장 이상이라면 당신은 사용자가 제출한 질문을 RAG 시스템에서 검색 효율을 높이기 위해 재작성하는 AI입니다.
    원본 질문: {original_query}
    재작성된 질문:"""
)













hallucination_prompt_template = PromptTemplate(
    input_variables=["documents", "generation"],
    template="""당신은 생성된 답변이 제공된 문서에서 정확히 유래했는지 평가하는 AI 평가자입니다.

다음은 문서와 생성된 답변입니다:

문서 (검색된 사실들):
{documents}

생성된 답변:
{generation}

평가 기준:
1. 생성된 답변이 제공된 문서에서 유래했는 경우 "yes"라고만 답하십시오.
2. 생성된 답변에 제공된 문서에 없는 내용이 포함되거나, 잘못된 정보가 포함된 경우 "no"라고만 답하십시오.
3. 어떠한 경우에도 추가적인 설명 없이 "yes" 또는 "no"로만 답변하십시오.
4. 응답이 "yes" 또는 "no" 외의 내용을 포함하면 평가가 무효 처리됩니다.

답변: 
"""
)

popup_store_verification_template = PromptTemplate(
    input_variables=["context", "query"],
    template=
    """
    후하게 판단하세요~~
    입력된 context와 query가 조금이라도 관련이 있다면 yes,
    전혀전혀 아예 관계가 없다면 no라고 답변하세요.

    context:
    {context}

    질문:
    {query}

    답변:
    """
)

popup_store_classification_template = PromptTemplate(
    input_variables=["query"],
    template=
    """
    당신은 질문의 주제를 분석하여 팝업스토어 혹은 온양 혹은 온당 혹은 이벤트 혹은 점검 와 관련된 질문인지 판단하는 AI입니다.
    -한 가지라도 해당하면 yes, 전부 해당하지 않을 경우에만 no라고 말하면 됩니다.
    질문이 팝업스토어 대여, 운영, 예약, 신청 방법 등 팝업스토어와 관련된 주제라면 "yes"라고 답변하세요.
    -한 가지라도 해당하면 yes, 전부 해당하지 않을 경우에만 no라고 말하면 됩니다.
    질문이 팝업스토어와 전혀 관련이 없는 주제라면 "no"라고 답변하세요.
    추가적인 설명 없이 "yes" 또는 "no"라고만 답변하세요.
    -응답이 "yes" 또는 "no" 외의 내용을 포함하면 평가가 무효 처리됩니다.

    질문:
    {query}

    답변:"""
)
