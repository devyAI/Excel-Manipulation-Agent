from __future__ import annotations

from typing import List
from .config import AzureSettings
from .azure_client import create_client


SYSTEM_PROMPT = (
    "You are a Python data wrangling assistant."
    " Output ONLY Python code that edits a provided Pandas DataFrame named df."
    " Do not include backticks, explanations, or print statements."
    " The code must be idempotent and safe."
)


def build_user_prompt(instruction: str, matched_columns: List[str]) -> str:
    cols_str = ", ".join(matched_columns)
    return (
        f"The DataFrame variable is named df. Relevant columns: [{cols_str}].\n"
        f"Instruction: {instruction}.\n"
        "Write Python Pandas code to perform the transformation on df."
    )


def generate_pandas_code(settings: AzureSettings, instruction: str, matched_columns: List[str]) -> str:
    client = create_client(settings)
    user_prompt = build_user_prompt(instruction, matched_columns)
    resp = client.chat.completions.create(
        model=settings.chat_deployment,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0,
    )
    code = resp.choices[0].message["content"].strip()
    return code


