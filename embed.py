#!/usr/bin/env python3
"""Generates embeddings via OpenAI or OpenRouter."""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_embedding_client():
    provider = os.getenv("EMBEDDING_PROVIDER", "openai")
    if provider == "openrouter":
        return OpenAI(
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        )
    return OpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    )

def embed(texts: list[str]) -> list[list[float]]:
    client = get_embedding_client()
    model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or "test embedding"
    result = embed([text])
    print(f"Embedding dim: {len(result[0])}")
    print(f"First 5 values: {result[0][:5]}")

