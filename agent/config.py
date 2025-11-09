from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv


@dataclass(frozen=True)
class AzureSettings:
    endpoint: str
    api_key: str
    chat_deployment: str
    embedding_deployment: str
    api_version: str = "2024-07-18"


def load_settings() -> AzureSettings:
    env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(dotenv_path=str(env_path))

    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    # Support either variable names for convenience
    chat_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT") or os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
    embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDINGS") or os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

    missing = []
    if not endpoint:
        missing.append("AZURE_OPENAI_ENDPOINT")
    if not api_key:
        missing.append("AZURE_OPENAI_API_KEY")
    if not chat_deployment:
        missing.append("AZURE_OPENAI_DEPLOYMENT or AZURE_OPENAI_CHAT_DEPLOYMENT")
    if not embedding_deployment:
        missing.append("AZURE_OPENAI_EMBEDDINGS or AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
    if missing:
        msg = (
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Checked .env at: {env_path}"
        )
        raise RuntimeError(msg)

    return AzureSettings(
        endpoint=endpoint,
        api_key=api_key,
        chat_deployment=chat_deployment,
        embedding_deployment=embedding_deployment,
    )


