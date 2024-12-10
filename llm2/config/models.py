from langchain_openai import ChatOpenAI
from .constants import DEFAULT_LLM_MODEL, TEMPERATURE, FAISS_PATH
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model=DEFAULT_LLM_MODEL, temperature=TEMPERATURE)
