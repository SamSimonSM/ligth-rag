import asyncio
import dotenv
from .rag_manager import RAGManager


dotenv.load_dotenv()

class BaseEmbeddingService:
    def __init__(self):
        self.rag = RAGManager.get_instance()
        

    async def tentar_insercao(self, texto: str, tentativas: int = 3, delay: float = 2.0):
        for i in range(tentativas):
            try:
                await self.rag.ainsert(texto)
                return True
            except Exception as e:
                print(f"Tentativa {i+1} falhou ao inserir. Erro: {e}")
                await asyncio.sleep(delay * (2 ** i))
        return False

    

    async def processar_documento(self):
        try:
            file_path = "..\\base.txt"
            with open(file_path, encoding='utf-8') as f:
                texto = f.read()
            sucesso = await self.tentar_insercao(texto) 
    
            if sucesso:
                print(f"Processado doc: {file_path}")
            else:
                print(f"Falha ao inserir embedding para doc: {file_path}")
        except Exception as e:
                print(f"Tentativa falhou. Erro: {e}")
                return e
        return sucesso
    