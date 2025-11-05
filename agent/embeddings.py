from __future__ import annotations

from typing import List
import numpy as np

from .config import AzureSettings
from .azure_client import create_client


def get_text_embedding(settings: AzureSettings, text: str) -> np.ndarray:
    client = create_client(settings)
    # Azure embeddings API expects input as a list of strings
    result = client.embeddings.create(
        model=settings.embedding_deployment,
        input=[text],
    )
    vec = result.data[0].embedding
    return np.array(vec, dtype=np.float32)


def get_text_embeddings(settings: AzureSettings, texts: List[str]) -> np.ndarray:
    client = create_client(settings)
    result = client.embeddings.create(
        model=settings.embedding_deployment,
        input=texts,
    )
    vectors = [item.embedding for item in result.data]
    return np.array(vectors, dtype=np.float32)


