import os
from lightrag import LightRAG
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, logger

class RAGManager:
    _instance = None

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LightRAG(
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
        return cls._instance

    @classmethod
    async def initialize(cls):
        instance = cls.get_instance()
        await instance.initialize_storages()
        await initialize_pipeline_status()
