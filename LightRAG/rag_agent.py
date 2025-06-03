"""Pydantic AI agent that leverages RAG with a local LightRAG for Pydantic documentation."""

import os
import sys
import argparse
from dataclasses import dataclass
import asyncio

import dotenv
from pydantic_ai import RunContext
from pydantic_ai.agent import Agent
from openai import AsyncOpenAI
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from lightrag import LightRAG, QueryParam
from lightrag.kg.shared_storage import initialize_pipeline_status

# Load environment variables from .env file
dotenv.load_dotenv()

WORKING_DIR = "./pydantic-docs"

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)



async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
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


    await rag.initialize_storages()

    return rag


@dataclass
class RAGDeps:
    """Dependencies for the RAG agent."""
    lightrag: LightRAG

ollama_model = OpenAIModel(
    model_name='qwen3:32b', provider=OpenAIProvider(base_url='https://llm.codeorbit.com.br/v1')
)

# Create the Pydantic AI agent
agent = Agent(
    ollama_model,
    deps_type=RAGDeps,
    system_prompt="You are a helpful assistant that answers questions about Pydantic AI based on the provided documentation. "
                  "Use the retrieve tool to get relevant information from the Pydantic AI documentation before answering. "
                  "If the documentation doesn't contain the answer, clearly state that the information isn't available "
                  "in the current documentation and provide your best general knowledge response.",
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
    return await context.deps.lightrag.aquery(
        search_query, param=QueryParam(mode="mix")
    )


async def run_rag_agent(question: str,) -> str:
    """Run the RAG agent to answer a question about Pydantic AI.
    
    Args:
        question: The question to answer.
        
    Returns:
        The agent's response.
    """
    # Create dependencies
    lightrag = await initialize_rag()
    deps = RAGDeps(lightrag=lightrag)
    
    # Run the agent
    result = await agent.run(question, deps=deps)
    
    return result.data


def main():
    """Main function to parse arguments and run the RAG agent."""
    parser = argparse.ArgumentParser(description="Run a Pydantic AI agent with RAG using ChromaDB")
    parser.add_argument("--question", help="The question to answer about Pydantic AI")
    
    args = parser.parse_args()
    
    # Run the agent
    response = asyncio.run(run_rag_agent(args.question))
    
    print("\nResponse:")
    print(response)


if __name__ == "__main__":
    main()
