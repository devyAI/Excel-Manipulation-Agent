from __future__ import annotations

import builtins as _builtins
import pandas as pd
import numpy as np
from typing import Tuple


class ExecutionError(RuntimeError):
    pass


def _restricted_builtins():
    allowed = {
        "True": True,
        "False": False,
        "None": None,
        "len": len,
        "range": range,
        "min": min,
        "max": max,
        "sum": sum,
        "any": any,
        "all": all,
        "abs": abs,
        "round": round,
        "enumerate": enumerate,
        "zip": zip,
        "sorted": sorted,
    }
    return allowed


def execute_pandas_code(df: pd.DataFrame, code: str) -> Tuple[pd.DataFrame, str]:
    safe_globals = {
        "__builtins__": _restricted_builtins(),
        "pd": pd,
        "np": np,
    }
    local_vars = {"df": df.copy()}

    try:
        compiled = compile(code, filename="<generated>", mode="exec")
        exec(compiled, safe_globals, local_vars)
    except Exception as exc:
        raise ExecutionError(f"Failed to execute generated code: {exc}")

    new_df = local_vars.get("df", None)
    if new_df is None or not isinstance(new_df, pd.DataFrame):
        raise ExecutionError("Generated code did not leave a DataFrame named df")

    # Simple change estimate (rows affected heuristic)
    summary = f"Rows: {len(df)} -> {len(new_df)}; Columns: {len(df.columns)}"
    return new_df, summary


