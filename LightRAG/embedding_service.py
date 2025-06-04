from pymongo import MongoClient
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.kg.shared_storage import initialize_pipeline_status
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug
import asyncio
import os
import dotenv

dotenv.load_dotenv()

class EmbeddingService:
    def __init__(self, mongo_uri, db_name, collection_name):
        # Configurações
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.collection_name = collection_name
        self.tamanho_lote = 100
        self.campo_flag = "ligthEmbedding"
        
        self.cliente_mongo = MongoClient(self.mongo_uri)
        self.colecao = self.cliente_mongo[self.db_name][self.collection_name]
        self.rag = LightRAG(
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
    
    
    def processar_documento(self, doc):
        try:
            """Processa um único documento, gerando seu embedding"""
            partes = [
                f"title: {doc.get('title', '')}",
                f"output: {doc.get('output', '')}",
                f"segment: {doc.get('segment', '')}",
                f"source: {doc.get('source', '')}",
                f"publishedDate: {doc.get('publishedDate', '')}",
                f"url: {doc.get('url', '')}",
                f"ticker: {doc.get('ticker', '')}",
                f"scrapedDate: {doc.get('scrapedDate', '')}"
            ]
            texto = " | ".join(partes).strip()
        
            if not texto:
                return None
            return texto

        except Exception as e:
            print(f"Erro ao salvar doc {doc['_id']}: {e}")
            return None
    
    async def processar_lote(self, ultimo_id=None):
        """Processa um lote de documentos"""
        filtro = {self.campo_flag: {"$in": [None, False]}}
        if ultimo_id:
            filtro["_id"] = {"$gt": ultimo_id}
        
        documentos = list(
            self.colecao.find(filtro)
                   .sort("_id", 1)
                   .limit(self.tamanho_lote)
        )
        
        if not documentos:
            return None
        
        print(f"Processando lote com {len(documentos)} documentos...")
        

        total_processados = 0
        
        for doc in documentos:
            try:
                texto = self.processar_documento(doc)
                if texto:
                    total_processados += 1
                
                    print(f"Processando documento {total_processados},{doc['_id']}")
                
                    await self.rag.ainsert(texto)
                
                    self.colecao.update_one(
                    {"_id": doc["_id"]},
                    {"$set": {self.campo_flag: True}}
                    )
                
                    print(f"Finalizado documento {total_processados},{doc['_id']}")
            except Exception as e:
                print(f"Erro ao salvar doc {doc['_id']}: {e}")
        return documentos[-1]["_id"], total_processados
    
    async def processar_todos(self):
        await self.rag.initialize_storages()
        await initialize_pipeline_status()
    
        ultimo_id = None
        total_processados = 0
        
        while True:
            ultimo_id_lote, processados_lote = await self.processar_lote(ultimo_id)
            
            if not ultimo_id_lote:
                print("Todos os documentos foram vetorizados e salvos no Qdrant.")
                break
            
            ultimo_id = ultimo_id_lote
            total_processados += processados_lote
            print(f"Lote finalizado. Total processados:{total_processados}")
        
        return total_processados
        