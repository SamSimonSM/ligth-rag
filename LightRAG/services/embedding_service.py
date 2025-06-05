from pymongo import MongoClient
from lightrag.utils import logger
import asyncio
import os
import dotenv
import time
from .rag_manager import RAGManager
from repositories.mongo_repository import MongoRepository

dotenv.load_dotenv()

class EmbeddingService:
    def __init__(self, mongo_repo: MongoRepository):
        self.mongo_repo = mongo_repo
        self.tamanho_lote = int(os.getenv("BATCH_SIZE", "100"))
        self.campo_flag = "ligthEmbedding"
        self.rag = RAGManager.get_instance()
        
    def processar_documento(self, doc: dict) -> str | None:
        try:
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
        documentos = self.mongo_repo.get_document_batch(
            self.campo_flag, self.tamanho_lote, ultimo_id
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
                self.mongo_repo.mark_as_processed(doc["_id"], self.campo_flag)
                total_processados += 1
                logger.info(f"Processado doc ID: {doc['_id']}")
            else:
                logger.error(f"Falha ao inserir embedding para doc {doc['_id']}")

        return documentos[-1]["_id"], total_processados

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