import asyncio
import os
from lightragDoc import QueryParam
import dotenv
from .rag_manager import RAGManager
from .rag_manager_gemini import RAGManagerGemini


dotenv.load_dotenv()

class SearchService:
    def __init__(self, llm=None):
        self.llm = llm
        if llm == "gemini":
            self.rag = RAGManagerGemini.get_instance()
        else:
            self.rag = RAGManager.get_instance()

    async def query(self, question: str, param: QueryParam):
        return await self.rag.aquery(
        question,
        param
    )
