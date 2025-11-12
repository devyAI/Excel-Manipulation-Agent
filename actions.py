"""Action functions for Excel data manipulation.

Each action function receives a DataFrame and action parameters,
and returns the modified DataFrame.
"""

import pandas as pd
import numpy as np
import streamlit as st
from typing import List, Dict, Any, Optional, Tuple
import re


def _get_row_indices(df: pd.DataFrame, row_range: Optional[Dict[str, int]]) -> slice:
    """Get row indices slice based on row_range.
    
    Args:
        df: Input DataFrame
        row_range: Optional dict with 'start' and 'end' keys, or None for all rows
        
    Returns:
        Slice object for row selection
    """
    if row_range is None:
        return slice(None)  # All rows
    return slice(row_range['start'], row_range['end'])


def set_value_action(df: pd.DataFrame,
                     target_columns,
                     parameters,
                     conditions=None,
                     row_range=None) -> pd.DataFrame:
    """
    Sets specific column values based on an optional condition.

    Args:
        df: Input DataFrame
        target_columns: List of column names to modify
        parameters: Dict containing 'value'
        conditions: Optional string condition (e.g., "FirstName-1 == 'Charlie'")
        row_range: Optional dict with 'start' and 'end' keys

    Returns:
        Updated DataFrame
    """
    result_df = df.copy()
    value = parameters.get("value", None)

    if value is None:
        print("⚠️ 'set_value' action missing 'value' parameter.")
        return result_df

    try:
        # If a condition is provided, create a boolean mask
        if conditions:
            mask = result_df.eval(conditions)
        else:
            mask = pd.Series(True, index=result_df.index)

        # Apply to row range if provided
        if row_range:
            start = row_range.get("start", 0)
            end = row_range.get("end", len(result_df))
            mask.iloc[:start] = False
            mask.iloc[end:] = False

        # Apply update
        for col in target_columns:
            if col not in result_df.columns:
                print(f"⚠️ Column '{col}' not found in DataFrame.")
                continue
            result_df.loc[mask, col] = value

        print(f"✅ Successfully updated {target_columns} where {conditions or 'all rows'}.")

    except Exception as e:
        print(f"❌ Error while applying condition '{conditions}': {e}")

    return result_df

def _make_formula_safe(expr: str, df: pd.DataFrame) -> str:
    """
    Replace plain column references (like Accidents-1) with df["Accidents-1"]
    so that eval() won't fail on column names with hyphens or spaces.
    """
    for col in df.columns:
        safe_col = re.escape(col)
        expr = re.sub(fr'\b{safe_col}\b', f'df["{col}"]', expr)
    return expr


def _fix_condition_syntax(expr: str) -> str:
    """
    Fixes assignment-like conditions (e.g. '=') to '==' for comparisons.
    Avoids replacing inside quotes or already correct expressions.
    """
    # Replace single = with == when not part of >=, <=, ==, or inside quotes
    def replacer(match):
        text = match.group(0)
        if '"' in text or "'" in text:
            return text
        return text.replace(" = ", " == ")
    
    expr = re.sub(r'(?<![<>=!])=(?!=)', '==', expr)  # replace lone =
    return expr


def update_action(
    df: pd.DataFrame, 
    target_columns: list, 
    parameters: dict, 
    conditions: str = None, 
    row_range: dict = None
) -> pd.DataFrame:
    """
    Enhanced update handler:
    - Supports both formula and value updates.
    - Handles '=' vs '==' automatically.
    - Works with conditional updates and row ranges.
    - Safely handles column names with spaces/hyphens.
    """
    result_df = df.copy()
    formula = parameters.get("formula")
    value = parameters.get("value")

    # Neither formula nor value provided
    if not formula and value is None:
        st.warning("No formula or value provided in parameters")
        return result_df

    # Prepare condition
    if conditions:
        conditions = _fix_condition_syntax(conditions)
        conditions = _make_formula_safe(conditions, result_df)

    # Handle row range if provided
    if row_range:
        start, end = row_range.get("start", 0), row_range.get("end", len(result_df))
        range_mask = pd.Series(False, index=result_df.index)
        range_mask.iloc[start:end] = True
    else:
        range_mask = pd.Series(True, index=result_df.index)

    for col in target_columns:
        if col not in result_df.columns:
            st.warning(f"Column '{col}' not found")
            continue

        try:
            # Create condition mask
            if conditions:
                try:
                    mask = eval(conditions, {"__builtins__": {}}, {'df': result_df, 'pd': pd, 'np': np})
                    if not isinstance(mask, pd.Series):
                        st.warning(f"Condition '{conditions}' did not produce a valid mask.")
                        continue
                except Exception as e:
                    st.error(f"Error evaluating condition '{conditions}': {str(e)}")
                    continue
            else:
                mask = pd.Series(True, index=result_df.index)

            mask = mask & range_mask

            if not mask.any():
                st.warning(f"No rows match conditions for column '{col}'")
                continue

            # Evaluation context
            context = {
                'df': result_df,
                'pd': pd,
                'np': np,
                'value': result_df.loc[mask, col],
            }

            if formula:
                safe_formula = _make_formula_safe(formula, result_df)
                try:
                    new_values = eval(safe_formula, {"__builtins__": {}}, context)
                    result_df.loc[mask, col] = new_values
                except Exception as e:
                    st.error(f"Error applying formula to column '{col}': {str(e)}")
            elif value is not None:
                # Auto-handle string literals
                if isinstance(value, str):
                    result_df.loc[mask, col] = value
                else:
                    result_df.loc[mask, col] = value

        except Exception as e:
            st.error(f"Error processing column '{col}': {str(e)}")
            continue

    return result_df

def filter_action(df: pd.DataFrame, target_columns: List[str],
                  parameters: Dict[str, Any], conditions: Optional[str] = None,
                  row_range: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Filter rows based on conditions.
    
    Args:
        df: Input DataFrame
        target_columns: List of column names to filter on
        parameters: Dict containing filter criteria (e.g., 'condition', 'value')
        conditions: Optional condition string
        row_range: Optional dict with 'start' and 'end' keys (note: filter applies to entire DataFrame)
        
    Returns:
        Filtered DataFrame
    """
    result_df = df.copy()
    
    # Note: For filter action, row_range is less relevant as filtering applies to entire DataFrame
    # But we can first filter by row_range if specified, then apply the filter condition
    
    # Use conditions parameter or parameters dict for filtering logic
    filter_condition = conditions or parameters.get('condition')
    filter_value = parameters.get('value')
    
    # If row_range is specified, we might want to only consider those rows for filtering
    # However, filtering typically applies to the entire dataset
    # For now, we'll apply the filter to the entire DataFrame regardless of row_range
    
    if filter_condition and filter_value:
        # Filter rows where any target column matches the condition
        mask = pd.Series([False] * len(result_df))
        for col in target_columns:
            if col in result_df.columns:
                mask |= (result_df[col] == filter_value)
        result_df = result_df[mask]
    elif filter_condition:
        # Use condition as a pandas query if possible
        try:
            result_df = result_df.query(filter_condition)
        except:
            st.warning(f"Could not parse filter condition: {filter_condition}")
    
    return result_df


def sum_action(df: pd.DataFrame, target_columns: List[str],
               parameters: Dict[str, Any], conditions: Optional[str] = None,
               row_range: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Create a new column with sum of target columns.
    
    Args:
        df: Input DataFrame
        target_columns: List of column names to sum
        parameters: Dict containing 'new_column' key for output column name
        conditions: Optional condition string (not used)
        row_range: Optional dict with 'start' and 'end' keys to specify row range
        
    Returns:
        DataFrame with new sum column
    """
    result_df = df.copy()
    new_col = parameters.get('new_column', 'sum')
    
    valid_columns = [col for col in target_columns if col in result_df.columns]
    
    if not valid_columns:
        st.warning("sum action requires at least one valid target column")
        return result_df
    
    # Get row slice
    row_slice = _get_row_indices(result_df, row_range)
    
    # Convert to numeric and sum
    numeric_cols = result_df.loc[row_slice, valid_columns].apply(pd.to_numeric, errors='coerce')
    
    # Initialize the new column if it doesn't exist
    if new_col not in result_df.columns:
        result_df[new_col] = None
    
    # Set sum values for the specified row range
    result_df.loc[row_slice, new_col] = numeric_cols.sum(axis=1)
    
    return result_df


def fill_random_action(df: pd.DataFrame, target_columns: List[str],
                       parameters: Dict[str, Any], conditions: Optional[str] = None,
                       row_range: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Fill target columns with random values.
    
    Supports:
    - Numeric ranges (min/max, integer or float)
    - Categorical values (yes/no, true/false, custom categories)
    - Boolean values
    - String values
    - Date values
    
    Args:
        df: Input DataFrame
        target_columns: List of column names to fill
        parameters: Dict with parameters:
            - 'min': minimum value for numeric ranges
            - 'max': maximum value for numeric ranges
            - 'data_type': 'numeric', 'integer', 'categorical', 'boolean', 'string', 'date'
            - 'categories': list of values for categorical data (e.g., ["yes", "no"])
            - 'integer_only': boolean for integer-only numeric values
        conditions: Optional condition string (not used)
        row_range: Optional dict with 'start' and 'end' keys to specify row range
        
    Returns:
        DataFrame with random values in target columns
    """
    result_df = df.copy()
    
    # Handle empty DataFrame
    if result_df.empty:
        num_rows = parameters.get('num_rows', 10)
        result_df = pd.DataFrame({col: [None] * num_rows for col in result_df.columns})
    
    # Get row slice
    row_slice = _get_row_indices(result_df, row_range)
    num_rows_to_fill = len(result_df.loc[row_slice]) if row_range else len(result_df)
    
    # Extract parameters
    data_type = parameters.get('data_type', '').lower()
    categories = parameters.get('categories', [])
    min_val = parameters.get('min')
    max_val = parameters.get('max')
    integer_only = parameters.get('integer_only', False)
    
    for col in target_columns:
        if col not in result_df.columns:
            continue
        
        # Get existing data for this column (for type inference)
        existing_data = result_df[col].dropna()
        
        # Determine data type if not explicitly provided
        col_data_type = data_type
        col_categories = categories.copy() if categories else []
        col_min_val = min_val
        col_max_val = max_val
        
        if not col_data_type and not col_categories:
            # Infer from column name or existing data
            col_lower = col.lower()
            
            # Check column name for hints
            if any(word in col_lower for word in ['status', 'active', 'enabled', 'flag', 'yes', 'no']):
                # Likely categorical or boolean
                if not col_categories:
                    col_categories = ['Yes', 'No']  # Default for status-like columns
                col_data_type = 'categorical'
            elif any(word in col_lower for word in ['age', 'price', 'amount', 'count', 'number', 'quantity', 'score']):
                # Likely numeric
                col_data_type = 'numeric'
                if col_min_val is None:
                    col_min_val = 0
                if col_max_val is None:
                    col_max_val = 100
            elif any(word in col_lower for word in ['date', 'time']):
                col_data_type = 'date'
            elif len(existing_data) > 0:
                # Infer from existing data
                if pd.api.types.is_bool_dtype(existing_data) or existing_data.dtype == 'bool':
                    col_data_type = 'boolean'
                    if not col_categories:
                        col_categories = [True, False]
                elif pd.api.types.is_numeric_dtype(existing_data):
                    col_data_type = 'numeric'
                    if col_min_val is None:
                        col_min_val = float(existing_data.min()) if len(existing_data) > 0 else 0
                    if col_max_val is None:
                        col_max_val = float(existing_data.max()) if len(existing_data) > 0 else 100
                else:
                    # Check if it's categorical with limited unique values
                    unique_vals = existing_data.unique()
                    if len(unique_vals) <= 10 and len(unique_vals) > 0:
                        col_data_type = 'categorical'
                        col_categories = unique_vals.tolist()
                    else:
                        col_data_type = 'string'
            else:
                # Default to numeric if no hints
                col_data_type = 'numeric'
                if col_min_val is None:
                    col_min_val = 0
                if col_max_val is None:
                    col_max_val = 100
        
        # Generate random values based on data type
        if col_data_type == 'categorical' or col_categories:
            # Categorical/Boolean values
            if not col_categories:
                # Default categories if none provided
                col_categories = ['Yes', 'No']
            
            # Ensure categories are appropriate type
            if len(existing_data) > 0 and not result_df[col].isna().all():
                # Try to match existing data type
                sample_val = existing_data.iloc[0]
                if isinstance(sample_val, bool):
                    col_categories = [bool(c) if isinstance(c, str) and c.lower() in ['true', 'false'] else c for c in col_categories]
                elif pd.api.types.is_numeric_dtype(existing_data):
                    try:
                        col_categories = [type(sample_val)(c) for c in col_categories]
                    except:
                        pass
            
            random_values = np.random.choice(col_categories, size=num_rows_to_fill)
        
        elif col_data_type == 'boolean' or (col_data_type == '' and not col_categories and col_min_val is None and col_max_val is None):
            # Boolean values
            random_values = np.random.choice([True, False], size=num_rows_to_fill)
        
        elif col_data_type == 'integer' or (col_data_type == 'numeric' and integer_only):
            # Integer values
            if col_min_val is None:
                col_min_val = 0
            if col_max_val is None:
                col_max_val = 100
            random_values = np.random.randint(int(col_min_val), int(col_max_val) + 1, size=num_rows_to_fill)
        
        elif col_data_type == 'numeric' or col_data_type == '':
            # Numeric (float) values
            if col_min_val is None:
                col_min_val = 0
            if col_max_val is None:
                col_max_val = 1
            random_values = np.random.uniform(float(col_min_val), float(col_max_val), size=num_rows_to_fill)
        
        elif col_data_type == 'date':
            # Date values
            try:
                from datetime import datetime, timedelta
                if col_min_val is None:
                    min_date = datetime(2020, 1, 1)
                else:
                    min_date = pd.to_datetime(col_min_val) if isinstance(col_min_val, str) else datetime(2020, 1, 1)
                
                if col_max_val is None:
                    max_date = datetime(2024, 12, 31)
                else:
                    max_date = pd.to_datetime(col_max_val) if isinstance(col_max_val, str) else datetime(2024, 12, 31)
                
                # Generate random dates
                time_between = (max_date - min_date).days
                random_days = np.random.randint(0, time_between + 1, size=num_rows_to_fill)
                random_values = [min_date + timedelta(days=int(d)) for d in random_days]
            except Exception as e:
                st.warning(f"Error generating dates for {col}: {str(e)}. Using default range.")
                random_values = pd.date_range('2020-01-01', periods=num_rows_to_fill, freq='D')
        
        elif col_data_type == 'string':
            # String values
            import string
            string_length = parameters.get('string_length', 10)
            random_values = [
                ''.join(np.random.choice(list(string.ascii_letters + string.digits), size=string_length))
                for _ in range(num_rows_to_fill)
            ]
        
        else:
            # Default: numeric
            if col_min_val is None:
                col_min_val = 0
            if col_max_val is None:
                col_max_val = 100
            random_values = np.random.uniform(float(col_min_val), float(col_max_val), size=num_rows_to_fill)
        
        # Assign values to the specified row range
        result_df.loc[row_slice, col] = random_values
    
    return result_df


def delete_action(df: pd.DataFrame, target_columns: List[str],
                  parameters: Dict[str, Any], conditions: Optional[str] = None,
                  row_range: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Delete target columns from DataFrame.
    
    Args:
        df: Input DataFrame
        target_columns: List of column names to delete
        parameters: Optional dict (not used)
        conditions: Optional condition string (not used)
        row_range: Optional dict with 'start' and 'end' keys (note: delete applies to entire columns)
        
    Returns:
        DataFrame with columns removed
    """
    result_df = df.copy()
    
    # Note: delete_action removes entire columns, so row_range is not applicable
    # But we'll keep the parameter for consistency with the interface
    
    valid_columns = [col for col in target_columns if col in result_df.columns]
    
    if valid_columns:
        result_df = result_df.drop(columns=valid_columns, errors='ignore')
    
    return result_df


def derive_column_action(df: pd.DataFrame, target_columns: List[str],
                         parameters: Dict[str, Any], conditions: Optional[str] = None,
                         row_range: Optional[Dict[str, int]] = None) -> pd.DataFrame:
    """Create a new derived column based on target columns.
    
    Args:
        df: Input DataFrame
        target_columns: List of column names to use for derivation
        parameters: Dict containing 'new_column' and 'formula' keys
        conditions: Optional condition string (not used)
        row_range: Optional dict with 'start' and 'end' keys to specify row range
        
    Returns:
        DataFrame with new derived column
    """
    result_df = df.copy()
    new_col = parameters.get('new_column')
    formula = parameters.get('formula')
    safe_formula = formula
    for col in target_columns:
        safe_formula = safe_formula.replace(col,f"df['{col}']")

    
    if not new_col:
        st.warning("derive_column action requires a 'new_column' parameter")
        return result_df
    
    if not formula:
        st.warning("derive_column action requires a 'formula' parameter")
        return result_df
    
    # Get row slice
    row_slice = _get_row_indices(result_df, row_range)
    
    try:
        # Evaluate formula with context including target columns
        context = {'df': result_df, 'pd': pd, 'np': np}
        for col in target_columns:
            if col in result_df.columns:
                context[col] = result_df[col]
        
        # Initialize the new column if it doesn't exist
        if new_col not in result_df.columns:
            result_df[new_col] = None
        
        # Evaluate formula - it should return a Series or array
        derived_values = eval(safe_formula, context)
        
        # Apply to the specified row range
        if isinstance(derived_values, pd.Series):
            result_df.loc[row_slice, new_col] = derived_values.loc[row_slice]
        else:
            # If it's an array-like, apply to the row range
            result_df.loc[row_slice, new_col] = derived_values[row_slice] if hasattr(derived_values, '__getitem__') else derived_values
    except Exception as e:
        st.error(f"Error deriving column: {str(e)}")
    
    return result_df


# Action registry mapping action types to functions
ACTION_REGISTRY = {
    'set_value': set_value_action,
    'update': update_action,
    'filter': filter_action,
    'sum': sum_action,
    'fill_random': fill_random_action,
    'delete': delete_action,
    'derive_column': derive_column_action,
}
