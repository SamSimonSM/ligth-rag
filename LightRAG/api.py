from fastapi import FastAPI, HTTPException
from services.embedding_service import EmbeddingService
import os
from dotenv import load_dotenv
from services.search_service import SearchService
from lightrag import QueryParam
from services.rag_manager import RAGManager
from repositories.mongo_repository import MongoRepository


load_dotenv()

# Get environment variables
MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
COLLECTION_NAME = os.getenv("COLLECTION_NAME")


app = FastAPI()

# Initialize services (consider dependency injection for larger apps)
mongo_repo = MongoRepository(MONGO_URI, DB_NAME, COLLECTION_NAME)
embedding_service = EmbeddingService(mongo_repo)
search_service = SearchService()

@app.on_event("startup")
async def startup_event():
    await RAGManager.initialize()

@app.get("/")
async def read_root():
    return {"message": "Financial Embedding API"}

@app.post("/embed")
async def run_embedding_process():
    try:
        total_processed = await embedding_service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")

@app.get("/search")
async def search_endpoint(question: str):
    try:
        result = await search_service.query(
            question,
            param=QueryParam(mode="mix")
        )
        return {"response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG query: {e}")