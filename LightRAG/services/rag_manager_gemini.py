import os
from lightragDoc import LightRAG
from lightragDoc.llm.hf import hf_embed
from lightragDoc.kg.shared_storage import initialize_pipeline_status
from lightragDoc.utils import EmbeddingFunc, setup_logger
from transformers import AutoModel, AutoTokenizer
from google import genai
from google.genai import types
from dotenv import load_dotenv

setup_logger("lightrag", level="INFO")
load_dotenv()


# PostgreSQL connection settings
os.environ["POSTGRES_HOST"] = os.getenv("POSTGRES_HOST", "147.79.104.83")
os.environ["POSTGRES_PORT"] = os.getenv("POSTGRES_PORT", "5432")
os.environ["POSTGRES_USER"] = os.getenv("POSTGRES_USER", "user")
os.environ["POSTGRES_PASSWORD"] = os.getenv("POSTGRES_PASSWORD", "postgres@2025")
os.environ["POSTGRES_DATABASE"] = os.getenv("POSTGRES_DATABASE", "ligthragdb")
gemini_api_key = os.getenv("GEMINI_API_KEY")
class RAGManagerGemini:
    _instance = None


    @staticmethod
    async def llm_model_func(
        prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
    ) -> str:
        # 1. Initialize the GenAI Client with your Gemini API Key
        client = genai.Client(api_key=gemini_api_key)
    
        # 2. Combine prompts: system prompt, history, and user prompt
        if history_messages is None:
            history_messages = []
    
        combined_prompt = ""
        if system_prompt:
            combined_prompt += f"{system_prompt}\n"
    
        for msg in history_messages:
            # Each msg is expected to be a dict: {"role": "...", "content": "..."}
            combined_prompt += f"{msg['role']}: {msg['content']}\n"
    
        # Finally, add the new user prompt
        combined_prompt += f"user: {prompt}"
    
        # 3. Call the Gemini model
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[combined_prompt],
            config=types.GenerateContentConfig(max_output_tokens=500, temperature=0.1),
        )
    
        # 4. Return the response text
        return response.text



    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = LightRAG(
                working_dir=os.getenv("WORKING_DIR", "data/"),
                llm_model_func=cls.llm_model_func,
                enable_llm_cache=False,
                enable_llm_cache_for_entity_extract=False,
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
                        )
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