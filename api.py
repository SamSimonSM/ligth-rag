from fastapi import FastAPI, HTTPException, Query
from lightRAG.services.news_embedding_service import NewsEmbeddingService
from lightRAG.services.fiis_embedding_service import FiisEmbeddingService
from lightRAG.services.fiis_details_embedding_service import FiisDetailsEmbeddingService
from lightRAG.services.base_embedding_service import BaseEmbeddingService
import os
from lightRAG.services.search_service import SearchService
from lightragDoc import QueryParam
from lightRAG.services.rag_manager import RAGManager
from lightRAG.services.rag_manager_gemini import RAGManagerGemini


app = FastAPI()

# Initialize service dictionaries to store instances for different LLM types
news_embedding_services = {}
fiis_embedding_services = {}
fiis_details_embedding_services = {}
base_embedding_services = {}
search_services = {}

@app.on_event("startup")
async def startup_event():
    # Initialize RAG managers
    await RAGManagerGemini.initialize()
    await RAGManager.initialize()
    
    # Initialize services with default LLM (None)
    news_embedding_services["default"] = NewsEmbeddingService(llm=None)
    fiis_embedding_services["default"] = FiisEmbeddingService(llm=None)
    fiis_details_embedding_services["default"] = FiisDetailsEmbeddingService(llm=None)
    base_embedding_services["default"] = BaseEmbeddingService(llm=None)
    search_services["default"] = SearchService(llm=None)
    
    # Initialize services with Gemini LLM
    news_embedding_services["gemini"] = NewsEmbeddingService(llm="gemini")
    fiis_embedding_services["gemini"] = FiisEmbeddingService(llm="gemini")
    fiis_details_embedding_services["gemini"] = FiisDetailsEmbeddingService(llm="gemini")
    base_embedding_services["gemini"] = BaseEmbeddingService(llm="gemini")
    search_services["gemini"] = SearchService(llm="gemini")

# Helper function to get the appropriate service based on LLM parameter
def get_service(service_dict, llm=None):
    if llm == "gemini":
        return service_dict["gemini"]
    return service_dict["default"]

@app.get("/")
async def read_root():
    return {"message": "Financial Embedding API"}

@app.post("/news-embed")
async def news_embedding_process(llm: str = Query(None, description="LLM model to use. Set to 'gemini' to use Gemini, or leave empty to use default")):
    try:
        # Get the appropriate pre-initialized service
        service = get_service(news_embedding_services, llm)
        total_processed = await service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed, "llm_used": llm or "default"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")
    
@app.post("/fiis-embed")
async def fiis_embedding_process(llm: str = Query(None, description="LLM model to use. Set to 'gemini' to use Gemini, or leave empty to use default")):
    try:
        # Get the appropriate pre-initialized service
        service = get_service(fiis_embedding_services, llm)
        total_processed = await service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed, "llm_used": llm or "default"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")
    

@app.post("/base-embed")
async def base_embedding_process(llm: str = Query(None, description="LLM model to use. Set to 'gemini' to use Gemini, or leave empty to use default")):
    try:
        # Get the appropriate pre-initialized service
        service = get_service(base_embedding_services, llm)
        total_processed = await service.processar_documento()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed, "llm_used": llm or "default"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")

@app.post("/fiis-details-embed")
async def fiis_details_embedding_process(llm: str = Query(None, description="LLM model to use. Set to 'gemini' to use Gemini, or leave empty to use default")):
    try:
        # Get the appropriate pre-initialized service
        service = get_service(fiis_details_embedding_services, llm)
        total_processed = await service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed, "llm_used": llm or "default"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")

@app.get("/search")
async def search_endpoint(question: str, llm: str = Query(None, description="LLM model to use. Set to 'gemini' to use Gemini, or leave empty to use default")):
    try:
        # Get the appropriate pre-initialized service
        service = get_service(search_services, llm)
        
        user_prompt = (
            "Responda apenas com informações verificáveis e baseadas em dados reais.\n"
            "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
            "\"Não tenho informações suficientes para responder com precisão.\"\n"
            "Não invente fatos, nomes, datas ou fontes.\n"
            "Quando possível, cite a fonte ou contexto da informação.\n"
            "Seja objetivo, factual e evite especulações."
        )
        
        # result = await service.query(
        #     question,
        #     param=QueryParam(mode="local", user_prompt=user_prompt)
        # )
        # result1 = await service.query(
        #     question,
        #     param=QueryParam(mode="global", user_prompt=user_prompt)
        # )
        # result2 = await service.query(
        #     question,
        #     param=QueryParam(mode="hybrid", user_prompt=user_prompt)
        # )
        # result3 = await service.query(
        #     question,
        #     param=QueryParam(mode="naive", user_prompt=user_prompt)
        # )
        result4 = await service.query(
            question,
            param=QueryParam(mode="mix", user_prompt=user_prompt)
        )
        return {"mix": result4, "llm_used": llm or "default"}
        # return {"local": result, "global":result1,"hybrid":result2, "naive": result3, "mix": result4, "llm_used": llm or "default"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG query: {e}")