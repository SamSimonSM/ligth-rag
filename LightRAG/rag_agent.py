"""Pydantic AI agent that leverages RAG with a local LightRAG for Pydantic documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
import asyncio

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from lightrag import LightRAG, QueryParam
from services.rag_manager import RAGManager

# Load environment variables from .env file
dotenv.load_dotenv()


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG

ollama_model = OpenAIModel(
    model_name='qwen3:14b', provider=OpenAIProvider(base_url='https://llm.codeorbit.com.br/v1')
)

# Create the Pydantic AI agent
agent = Agent(
    ollama_model,
    deps_type=RAGDeps,
    system_prompt="Você atua como Cody Felipe, consultor da Code FIIs, empresa especializada em Fundos de Investimento Imobiliário no Brasil. "
               "Use a ferramenta de retrieve para obter informações relevantes sobre FIIs (Fundos de Investimento Imobiliário) antes de responder. "
               "Se a documentação não contiver a resposta, declare claramente que a informação não está disponível na documentação atual "
               "e forneça sua melhor resposta com base em conhecimento técnico confiável. "
               "Foque exclusivamente no mercado financeiro brasileiro e em FIIs, evitando temas fora desse escopo. "
               "Prefira análises comparativas, critérios objetivos e dados como Dividend Yield, P/VP, Vacância, Cap Rate, IFIX, CDI, Selic, entre outros. "
               "Utilize linguagem técnica e acessível, com precisão e clareza. "
               "Considere sempre o cenário regulatório brasileiro, como CVM e B3. "
               "As respostas devem ter no máximo 500 caracteres. "
               "Nunca mencione que é uma inteligência artificial, ferramenta ou sistema automatizado. "
               "Você responde sempre como um consultor humano da Code FIIs, com objetividade e conhecimento técnico. "
               "Sua resposta deve conter apenas o conteúdo final da análise, sem introduções ou observações sobre o sistema."
               "Responda sempre em Português.",
)


@agent.tool
async def retrieve(context: RunContext[RAGDeps], search_query: str) -> str:
    """Retrieve relevant documents from ChromaDB based on a search query.
    
    Args:
        context: The run context containing dependencies.
        search_query: The search query to find relevant documents.
        
    Returns:
        Formatted context information from the retrieved documents.
    """
    text = await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="mix")
    )
    if "</think>" in text:
        conteudo_pos_think = text.split("</think>")[-1].strip()
    else:
        conteudo_pos_think = text.strip()

    return conteudo_pos_think


async def run_rag_agent(question: str,) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = RAGManager.get_instance()
    deps = RAGDeps(lightrag=lightrag)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.data


def main():
    """Main function to executar o RAG agent com pergunta fixa."""
    pergunta_fixa = "Quais melhores fundos de investimento imobiliario para investir atualmente?"

    asyncio.run(RAGManager.initialize())

    response = asyncio.run(run_rag_agent(pergunta_fixa))

    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
