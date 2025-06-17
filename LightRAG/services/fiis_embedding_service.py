from pymongo import MongoClient
import asyncio
import os
import dotenv
import time
from .rag_manager import RAGManager
from repositories.mongo_repository import MongoRepository




dotenv.load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
FIIS_COLLECTION_NAME = os.getenv("FIIS_COLLECTION_NAME")

mongo_repo = MongoRepository(MONGO_URI, DB_NAME, FIIS_COLLECTION_NAME)

class FiisEmbeddingService:
    def __init__(self):
        self.mongo_repo = mongo_repo
        self.tamanho_lote = int(os.getenv("BATCH_SIZE", "100"))
        self.rag = RAGManager.get_instance()
        self.campo_flag = "ligthEmbedding"

        
    def processar_documento(self, doc: dict) -> str | None:
        try:
            partes = [f"{k}: {v}" for k, v in doc.items()]
            texto = " | ".join(partes).strip()
            return texto if texto else None
        except Exception as e:
            print(f"Erro ao processar documento ID {doc.get('_id')}: {e}")
            return None

    async def tentar_insercao(self, texto: str, tentativas: int = 3, delay: float = 2.0):
        for i in range(tentativas):
            try:
                await self.rag.ainsert(texto)
                return True
            except Exception as e:
                print(f"Tentativa {i+1} falhou ao inserir. Erro: {e}")
                await asyncio.sleep(delay * (2 ** i))
        return False

    async def processar_lote(self, ultimo_id=None):
        documentos = self.mongo_repo.get_document_active_batch(
            self.tamanho_lote, ultimo_id
        )

        if not documentos:
            return None, 0

        total_processados = 0

        for doc in documentos:
            texto = self.processar_documento(doc)
            if not texto:
                continue

            sucesso = await self.tentar_insercao(texto)
            if sucesso:
                total_processados += 1
                print(f"Processado doc ID: {doc['_id']}")
                self.mongo_repo.mark_as_processed(doc["_id"], self.campo_flag)
            else:
                print(f"Falha ao inserir embedding para doc {doc['_id']}")

        return documentos[-1]["_id"], total_processados

    async def processar_todos(self):
        ultimo_id = None
        total_processados = 0

        while True:
            ultimo_id_lote, processados_lote = await self.processar_lote(ultimo_id)

            if not ultimo_id_lote:
                print("Todos os documentos foram vetorizados.")
                break

            ultimo_id = ultimo_id_lote
            total_processados += processados_lote
            print(f"Lote finalizado. Total processados: {total_processados}")

        return total_processados