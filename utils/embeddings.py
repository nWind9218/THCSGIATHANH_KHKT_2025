"""Embedding and vector utilities for pgvector operations."""

import asyncio
import os
from typing import Optional

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

_embeddings: Optional[OpenAIEmbeddings] = None


def get_embeddings() -> OpenAIEmbeddings:
    """Singleton embeddings instance."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            api_key=os.getenv("OPENAI_API_EMBEDDING_KEY") or os.getenv("OPENAI_API_KEY"),
        )
    return _embeddings


async def embed_text(text: str) -> list[float]:
    """Embed text using OpenAI embeddings."""
    loop = asyncio.get_running_loop()
    emb = get_embeddings()
    return await loop.run_in_executor(None, emb.embed_query, text)


def vector_literal(vector: list[float]) -> str:
    """Convert float vector to pgvector literal format."""
    return "[" + ",".join(str(x) for x in vector) + "]"
