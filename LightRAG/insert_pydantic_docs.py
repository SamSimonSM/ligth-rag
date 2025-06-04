import os
import asyncio
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
import dotenv
import httpx
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
from pymongo import MongoClient

# Load environment variables from .env file
dotenv.load_dotenv()

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)
    
# URL of the Pydantic AI documentation
PYDANTIC_DOCS_URL = "https://ai.pydantic.dev/llms.txt"


def processar_lote(self, ultimo_id=None):
        """Processa um lote de documentos"""
        filtro = {self.campo_flag: {"$in": [None, False]}}
        if ultimo_id:
            filtro["_id"] = {"$gt": ultimo_id}
        
        documentos = list(
            self.colecao.find(filtro)
                   .sort("_id", 1)
        )
        
        if not documentos:
            return None, 0
             
        return documentos

def fetch_pydantic_docs() -> str:
    """Fetch the Pydantic AI documentation from the URL.
    
    Returns:
        The content of the documentation
    """
    try:
        response = httpx.get(PYDANTIC_DOCS_URL)
        response.raise_for_status()
        return response.text
    except Exception as e:
        raise Exception(f"Error fetching Pydantic AI documentation: {e}")


async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
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

    return rag


def main():
    # Initialize RAG instance and insert Pydantic documentation
    rag = asyncio.run(initialize_rag())
    rag.insert(fetch_pydantic_docs())

if __name__ == "__main__":
    main()