## Project: Starbucks RAG Decision Intelligence Platform

This is a production RAG platform. Business users ask natural language questions about Starbucks operational data and get AI-powered answers.

## Conda Environment
Always use: conda activate starbucks-rag

## Tech Stack
- Python 3.11, FastAPI (async), Pydantic v2
- LangChain + LangGraph for orchestration
- Sentence-transformers (all-MiniLM-L6-v2) for embeddings — local, no API key
- FAISS for local vector store
- OpenAI GPT-3.5 locally (switch to AWS Bedrock via LLM_PROVIDER=bedrock)
- Streamlit for frontend UI

## Key Entry Points
- Generate data: python data/synthetic/generate_data.py
- Build index: python scripts/build_index.py
- Start API: uvicorn api.main:app --reload --port 8000
- Start UI: streamlit run ui/app.py --server.port 8501
- Run tests: pytest tests/ -v --cov=.

## Code Rules
- ALL functions must have type hints
- ALL classes and public methods must have docstrings
- Use async/await in all FastAPI routes
- All request/response schemas use Pydantic BaseModel
- Use Python logging module — never use print() in production code
- Error handling: always wrap external calls in try/except

## Architecture Flow
User Query → FastAPI POST /query → HybridRetriever (FAISS + BM25) → LangGraph Agent → RAGChain → LLM → GuardrailsValidator → JSON Response → Streamlit UI
