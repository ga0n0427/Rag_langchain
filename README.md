# **LangChain-Based Document Search and Q&A System**
This project leverages **LangChain** to build a document-based search and Q&A system. It supports efficient document retrieval, LLM-driven response generation, hallucination verification, and real-time database synchronization.
## **Features**
- **PostgreSQL Integration**  
  Listens to PostgreSQL notifications and retrieves updated values automatically. Ensures real-time synchronization with the database.
- **Vector Database Support**  
  Stores and retrieves embeddings in a vector database (FAISS). Enables fast and accurate similarity-based document retrieval.
- **Document Processing**  
  Uses TikToken for token-based input control and semantic chunking to differentiate text meaningfully.
- **Query Workflow**  
  - Rewrites user queries for improved search efficiency.
  - Classifies related questions for better context understanding.
  - Generates accurate responses based strictly on retrieved content.
  - Verifies answers to prevent hallucination or misinformation.
- **Cohere API-Based Re-Ranking**  
  Utilizes Cohere's API to re-rank search results for enhanced relevance and accuracy.
## **Environment Setup and Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/langchain-qa-system.git
   cd langchain-qa-system
2. Set up a Python environment:
   python -m venv env
  source env/bin/activate  # Linux/Mac
  env\Scripts\activate     # Windows
3. Install dependencies:
   pip install -r requirements.txt
4. Set up PostgreSQL:
    Configure your PostgreSQL database to send notifications.
5. Run the application:
   python app.py
# **LangChain 기반 문서 검색 및 Q&A 시스템**
이 프로젝트는 **LangChain**을 활용하여 문서 기반 검색 및 Q&A 시스템을 구축합니다. 효율적인 문서 검색, LLM 기반 답변 생성, 할루시네이션 검증, 실시간 데이터베이스 동기화를 지원합니다.
## **주요 기능**
- **PostgreSQL 통합**  
  PostgreSQL 알림을 수신하여 자동으로 최신 데이터를 가져옵니다. 데이터베이스와 실시간 동기화를 보장합니다.
- **벡터 데이터베이스 지원**  
  FAISS를 활용하여 벡터 임베딩을 저장하고 검색합니다. 빠르고 정확한 유사성 기반 문서 검색을 지원합니다.
- **문서 처리**  
  TikToken을 사용하여 입력 크기를 제어하고, 의미를 기반으로 텍스트를 구분하는 시맨틱 청킹을 구현합니다.
- **쿼리 워크플로**  
  - 사용자 질문을 검색 효율을 높이도록 재작성합니다.
  - 관련 질문을 분류하여 문맥을 더욱 잘 이해합니다.
  - 검색된 콘텐츠를 기반으로 정확한 답변을 생성합니다.
  - 생성된 답변이 근거 없는 정보나 할루시네이션을 포함하지 않도록 검증합니다.
- **Cohere API 기반 리랭크**  
  Cohere API를 활용하여 검색 결과를 재정렬하고, 더욱 관련성 높은 문서를 우선적으로 제공합니다.
