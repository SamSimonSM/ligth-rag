import asyncio
import os
from embedding_service import EmbeddingService
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import dotenv

dotenv.load_dotenv()

async def main():

    MONGO_URI = os.getenv("MONGO_URI")
    DB_NAME = os.getenv("DB_NAME")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME")
    embedding_service = EmbeddingService(MONGO_URI, DB_NAME, COLLECTION_NAME)

    rag = LightRAG(
        working_dir="data/",
        llm_model_func=ollama_model_complete,
        llm_model_name=os.getenv("LLM_MODEL", "qwen2.5:14b"),
        llm_model_max_token_size=8192,
        llm_model_kwargs={
            "host": os.getenv("LLM_BINDING_HOST", "https://llm.codeorbit.com.br"),
            "options": {"num_ctx": 8192},
            "timeout": int(os.getenv("TIMEOUT", "300")),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=lambda texts: ollama_embed(
                texts,
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "https://llm.codeorbit.com.br"),
            ),
        ),
    )

    await rag.initialize_storages()
    await initialize_pipeline_status()
    
    documentos = await embedding_service.processar_todos()
 
    result = await rag.aquery(
        "quais os melhores fiis para investir?",
        param=QueryParam(mode="mix")
    )

    print(result)

if __name__ == "__main__":
    asyncio.run(main())