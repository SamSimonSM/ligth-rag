import os
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.llm.hf import hf_model_complete, hf_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, setup_logger
from transformers import AutoModel, AutoTokenizer
from dotenv import load_dotenv

setup_logger("lightrag", level="INFO")
load_dotenv()


# PostgreSQL connection settings
os.environ["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "147.79.104.83")
os.environ["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5432")
os.environ["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "user")
os.environ["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "postgres@2025")
os.environ["POSTGRES_DATABASE"] = os.getenv("POSTGRES_DATABASE", "ligthragdb")

class RAGManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LightRAG(
                working_dir=os.getenv("WORKING_DIR", "data/"),
                llm_model_func=ollama_model_complete,
                llm_model_name=os.getenv("LLM_MODEL", "cogito:32b"),
                llm_model_max_token_size=int(os.getenv("LLM_MAX_TOKEN_SIZE", "8192")),
                llm_model_kwargs={
                    "host": os.getenv("LLM_BINDING_HOST", "https://llm.codeorbit.com.br"),
                    "options": {"num_ctx": int(os.getenv("LLM_NUM_CTX", "8192"))},
                    "timeout": int(os.getenv("TIMEOUT", "10000")),
                },
                embedding_func=EmbeddingFunc(
                    embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
                    max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
                    func=lambda texts: hf_embed(
                        texts,
                        tokenizer=AutoTokenizer.from_pretrained(
                            os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
                        ),
                        embed_model=AutoModel.from_pretrained(
                            os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
                        ),
                    # func=lambda texts: ollama_embed(
                    #     texts,
                    #     embed_model="bge-m3:latest"
                    )
                ),
                kv_storage=os.getenv("KV_STORAGE", "PGKVStorage"),
                doc_status_storage=os.getenv("DOC_STATUS_STORAGE", "PGDocStatusStorage"),
                graph_storage=os.getenv("GRAPH_STORAGE", "PGGraphStorage"),
                vector_storage=os.getenv("VECTOR_STORAGE", "PGVectorStorage"),
                auto_manage_storages_states=os.getenv("AUTO_MANAGE_STORAGES_STATES", "False"),
            )
        return cls._instance

    @classmethod
    async def initialize(cls):
        instance = cls.get_instance()
        
        try:
            await instance.initialize_storages()
            await initialize_pipeline_status()
        except Exception as e:
                print(f"Erro: {e}")