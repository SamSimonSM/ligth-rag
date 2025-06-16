from fastapi import FastAPI, HTTPException
from services.news_embedding_service import NewsEmbeddingService
from services.fiis_embedding_service import FiisEmbeddingService
from services.fiis_details_embedding_service import FiisDetailsEmbeddingService
from services.base_embedding_service import BaseEmbeddingService
import os
from services.search_service import SearchService
from lightrag import QueryParam
from services.rag_manager import RAGManager
from repositories.mongo_repository import MongoRepository


app = FastAPI()

# Initialize services (consider dependency injection for larger apps)
news_embedding_service = NewsEmbeddingService()
fiis_embedding_service = FiisEmbeddingService()
fiis_details_embedding_service = FiisDetailsEmbeddingService()
base_embedding_service=BaseEmbeddingService()
search_service = SearchService()

@app.on_event("startup")
async def startup_event():
    await RAGManager.initialize()

@app.get("/")
async def read_root():
    return {"message": "Financial Embedding API"}

@app.post("/news-embed")
async def news_embedding_process():
    try:
        total_processed = await news_embedding_service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")
    
@app.post("/fiis-embed")
async def fiis_embedding_process():
    try:
        total_processed = await fiis_embedding_service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")
    

@app.post("/base-embed")
async def fiis_embedding_process():
    try:
        total_processed = await base_embedding_service.processar_documento()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")

@app.post("/fiis-details-embed")
async def fiis_details_embedding_process():
    try:
        total_processed = await fiis_details_embedding_service.processar_todos()
        return {"message": "Embedding process completed", "total_documents_processed": total_processed}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during embedding process: {e}")

@app.get("/search")
async def search_endpoint(question: str):
    try:
        result = await search_service.query(
            question,
            param=QueryParam(mode="local",
            user_prompt = 
                ("Responda apenas com informações verificáveis e baseadas em dados reais.\n"
                "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
                "\"Não tenho informações suficientes para responder com precisão.\"\n"
                "Não invente fatos, nomes, datas ou fontes.\n"
                "Quando possível, cite a fonte ou contexto da informação.\n"
                "Seja objetivo, factual e evite especulações."
                )
            )
        )
        # result1 = await search_service.query(
        #     question,
        #     param=QueryParam(mode="global",
        #         user_prompt = 
        #         ("Responda apenas com informações verificáveis e baseadas em dados reais.\n"
        #         "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
        #         "\"Não tenho informações suficientes para responder com precisão.\"\n"
        #         "Não invente fatos, nomes, datas ou fontes.\n"
        #         "Quando possível, cite a fonte ou contexto da informação.\n"
        #         "Seja objetivo, factual e evite especulações."
        #         )
        #     )
        # )
        # result2 = await search_service.query(
        #     question,
        #     param=QueryParam(mode="hybrid",
        #         user_prompt = 
        #         ("Responda apenas com informações verificáveis e baseadas em dados reais.\n"
        #         "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
        #         "\"Não tenho informações suficientes para responder com precisão.\"\n"
        #         "Não invente fatos, nomes, datas ou fontes.\n"
        #         "Quando possível, cite a fonte ou contexto da informação.\n"
        #         "Seja objetivo, factual e evite especulações."
        #         )
        #     )
        # )
        result3 = await search_service.query(
            question,
            param=QueryParam(mode="naive",
                user_prompt = 
                ("Responda apenas com informações verificáveis e baseadas em dados reais.\n"
                "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
                "\"Não tenho informações suficientes para responder com precisão.\"\n"
                "Não invente fatos, nomes, datas ou fontes.\n"
                "Quando possível, cite a fonte ou contexto da informação.\n"
                "Seja objetivo, factual e evite especulações."
                )
            )
        )
        result4 = await search_service.query(
            question,
            param=QueryParam(
                mode="mix",
                user_prompt = 
                ("Responda apenas com informações verificáveis e baseadas em dados reais.\n"
                "Se você não souber a resposta ou não tiver certeza, diga claramente: \"Não sei\" ou "
                "\"Não tenho informações suficientes para responder com precisão.\"\n"
                "Não invente fatos, nomes, datas ou fontes.\n"
                "Quando possível, cite a fonte ou contexto da informação.\n"
                "Seja objetivo, factual e evite especulações."
                )
            )
        )
        return {"local": result, "naive": result3, "mix": result4}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during RAG query: {e}")