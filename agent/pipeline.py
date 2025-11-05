from __future__ import annotations

from typing import List, Tuple
import pandas as pd
import numpy as np

from .config import load_settings
from .column_matcher import rank_columns
from .codegen import generate_pandas_code
from .executor import execute_pandas_code


def read_excel_headers(path: str) -> List[str]:
    df = pd.read_excel(path)
    return [str(c) for c in df.columns]


def run_transformation(input_excel_path: str, instruction: str, output_excel_path: str | None = None) -> Tuple[str, str]:
    settings = load_settings()

    df = pd.read_excel(input_excel_path)
    # Understanding stage: robust column ranking using names, sample values and lexical signal
    matched_cols, scores = rank_columns(settings, df, instruction, k=min(3, len(df.columns)))

    # Action stage: generate code and execute
    code = generate_pandas_code(settings, instruction, matched_cols)
    new_df, exec_summary = execute_pandas_code(df, code)

    # Output stage: save file and build summary
    if output_excel_path is None:
        output_excel_path = _default_output_path(input_excel_path)
    new_df.to_excel(output_excel_path, index=False)

    summary = (
        f"Applied instruction on columns: {matched_cols}. "
        f"Scores: {[round(float(s), 3) for s in scores]}. "
        f"{exec_summary}"
    )
    return output_excel_path, summary


def _default_output_path(input_path: str) -> str:
    import os
    base, ext = os.path.splitext(input_path)
    return f"{base}_updated{ext or '.xlsx'}"


