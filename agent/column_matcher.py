from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np
from difflib import SequenceMatcher

from .config import AzureSettings
from .embeddings import get_text_embedding, get_text_embeddings
from .similarity import cosine_similarity_matrix


def _sample_values_as_text(series: pd.Series, max_unique: int = 20) -> str:
    # Take up to N unique non-null samples, as strings
    samples = list(dict.fromkeys([str(v) for v in series.dropna().tolist()]))[:max_unique]
    return ", ".join(samples)


def _lexical_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def rank_columns(settings: AzureSettings, df: pd.DataFrame, instruction: str, k: int = 3) -> Tuple[List[str], List[float]]:
    headers = [str(c) for c in df.columns]

    # Build two contexts per column: header name and sample values
    header_texts = headers
    value_texts = [_sample_values_as_text(df[col]) for col in df.columns]

    # Embeddings
    header_vecs = get_text_embeddings(settings, header_texts)
    value_vecs = get_text_embeddings(settings, value_texts)
    instr_vec = get_text_embedding(settings, instruction)

    # Similarities
    sim_header = cosine_similarity_matrix(instr_vec, header_vecs)[0]
    sim_values = cosine_similarity_matrix(instr_vec, value_vecs)[0]
    sim_lex = np.array([_lexical_similarity(instruction, h) for h in headers], dtype=np.float32)

    # Weighted composite score
    # Higher weight on semantic header name, then values, small lexical tie-breaker
    score = 0.6 * sim_header + 0.35 * sim_values + 0.05 * sim_lex

    idx = np.argsort(-score)[: max(1, min(k, len(headers)))]
    return [headers[i] for i in idx], [float(score[i]) for i in idx]


