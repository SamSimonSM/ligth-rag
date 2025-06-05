import asyncio
import os
from lightrag import LightRAG, QueryParam
import dotenv
from .rag_manager import RAGManager


dotenv.load_dotenv()

class SearchService:
    def __init__(self):
        self.rag = RAGManager.get_instance()

    async def query(self, question: str, param: QueryParam):
        return await self.rag.aquery(
        "quais os melhores fiis para investir?",
        param=QueryParam(mode="mix")
    )
