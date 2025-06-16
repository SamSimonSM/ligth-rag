from pymongo import MongoClient
from lightrag.utils import logger
import asyncio
import os
import dotenv
import time
from .rag_manager import RAGManager
from repositories.mongo_repository import MongoRepository




dotenv.load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
DB_NAME = os.getenv("DB_NAME")
DETAILS_COLLECTION_NAME = os.getenv("DETAILS_COLLECTION_NAME")

mongo_repo = MongoRepository(MONGO_URI, DB_NAME, DETAILS_COLLECTION_NAME)

class FiisDetailsEmbeddingService:
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
            logger.error(f"Erro ao processar documento ID {doc.get('_id')}: {e}")
            return None

    async def tentar_insercao(self, texto: str, tentativas: int = 3, delay: float = 2.0):
        for i in range(tentativas):
            try:
                await self.rag.ainsert(texto)
                return True
            except Exception as e:
                logger.warning(f"Tentativa {i+1} falhou ao inserir. Erro: {e}")
                await asyncio.sleep(delay * (2 ** i))
        return False

    async def processar_lote(self, ultimo_id=None):
        documentos = self.mongo_repo.get_document_active_batch(
             self.tamanho_lote, ultimo_id, self.campo_flag
        )

        if not documentos:
            return None, 0

        total_processados = 0

        for doc in documentos:
            texto = self.processar_documento(doc)
            if not texto:
                continue

            sucesso = await self.tentar_insercao(texto)
            # await self.merge_entities(doc)
            
            if sucesso:
                total_processados += 1
                logger.info(f"Processado doc ID: {doc['_id']}")
                self.mongo_repo.mark_as_processed(doc["_id"], self.campo_flag)
            else:
                logger.error(f"Falha ao inserir embedding para doc {doc['_id']}")

        return documentos[-1]["_id"], total_processados
    
    
    async def merge_entities(self, doc: dict) -> bool | None:
        try:
            ticker = doc.get('ticker', '')
            nome = doc.get('nome', '')

            for i in range(3):  # Tentativas com backoff exponencial
                try:
                    await self.rag.amerge_entities(
                        source_entities=[ticker, nome],
                        target_entity=ticker
                    )
                    logger.info(f"Entidades mescladas com sucesso para ticker: {ticker}")
                    return True
                except Exception as e:
                    logger.warning(f"Tentativa {i+1} falhou ao mesclar entidades. Erro: {e}")
                    await asyncio.sleep(2.0 * (2 ** i))  # Espera crescente: 2s, 4s, 8s
            return None  # Se falhar todas as tentativas
        except Exception as e:
            logger.error(f"Erro inesperado ao tentar mesclar entidades: {e}")
            return None

   
   
    async def processar_todos(self):
        ultimo_id = None
        total_processados = 0

        while True:
            ultimo_id_lote, processados_lote = await self.processar_lote(ultimo_id)

            if not ultimo_id_lote:
                logger.info("Todos os documentos foram vetorizados.")
                break

            ultimo_id = ultimo_id_lote
            total_processados += processados_lote
            logger.info(f"Lote finalizado. Total processados: {total_processados}")

        return total_processados