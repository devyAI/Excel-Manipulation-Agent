from __future__ import annotations

from openai import AzureOpenAI
from .config import AzureSettings


def create_client(settings: AzureSettings) -> AzureOpenAI:
    return AzureOpenAI(
        azure_endpoint=settings.endpoint,
        api_key=settings.api_key,
        api_version=settings.api_version,
    )


