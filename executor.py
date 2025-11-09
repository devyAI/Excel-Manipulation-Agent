"""JSON Action Executor for Excel transformations.

This module parses JSON action schemas and executes the corresponding actions.
"""

import pandas as pd
import streamlit as st
from typing import Dict, Any, List, Optional, Tuple
from actions import ACTION_REGISTRY


class ActionSchema:
    """Schema definition for action JSON structure."""
    
    REQUIRED_ACTION_FIELDS = ['type', 'target_columns']
    OPTIONAL_ACTION_FIELDS = ['conditions', 'parameters', 'row_range']
    
    VALID_ACTION_TYPES = [
        'set_value',
        'update',
        'filter',
        'sum',
        'fill_random',
        'delete',
        'derive_column'
    ]
    
    @staticmethod
    def validate_action(action: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate an action against the schema.
        
        Args:
            action: Action dictionary to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        for field in ActionSchema.REQUIRED_ACTION_FIELDS:
            if field not in action:
                return False, f"Missing required field: {field}"
        
        # Validate action type
        action_type = action.get('type', '').lower()
        if action_type not in ActionSchema.VALID_ACTION_TYPES:
            return False, f"Invalid action type: {action_type}. Must be one of {ActionSchema.VALID_ACTION_TYPES}"
        
        # Validate target_columns is a list
        target_columns = action.get('target_columns', [])
        if not isinstance(target_columns, list):
            return False, "target_columns must be a list"
        
        if not target_columns:
            return False, "target_columns cannot be empty"
        
        # Validate parameters is a dict if present
        if 'parameters' in action and not isinstance(action['parameters'], dict):
            return False, "parameters must be a dictionary"
        
        # Validate conditions is a string if present
        if 'conditions' in action and not isinstance(action['conditions'], str):
            return False, "conditions must be a string"
        
        # Validate row_range is a dict with start and end if present
        if 'row_range' in action:
            row_range = action['row_range']
            if not isinstance(row_range, dict):
                return False, "row_range must be a dictionary"
            if 'start' not in row_range or 'end' not in row_range:
                return False, "row_range must contain 'start' and 'end' keys"
            if not isinstance(row_range['start'], int) or not isinstance(row_range['end'], int):
                return False, "row_range start and end must be integers"
            if row_range['start'] < 0:
                return False, "row_range start must be non-negative"
            if row_range['end'] <= row_range['start']:
                return False, "row_range end must be greater than start"
        
        return True, None


class ActionExecutor:
    """Executor that parses JSON and executes actions."""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize executor with a DataFrame.
        
        Args:
            df: Input DataFrame to transform
        """
        self.df = df.copy()
        self.result_df = df.copy()
    
    def validate_json_schema(self, json_data: Dict[str, Any]) -> Tuple[bool, Optional[str]]:
        """Validate the entire JSON schema.
        
        Args:
            json_data: JSON data containing actions
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not isinstance(json_data, dict):
            return False, "JSON data must be a dictionary"
        
        if 'actions' not in json_data:
            return False, "JSON data must contain 'actions' key"
        
        if not isinstance(json_data['actions'], list):
            return False, "'actions' must be a list"
        
        if not json_data['actions']:
            return False, "'actions' list cannot be empty"
        
        # Validate each action
        for i, action in enumerate(json_data['actions']):
            is_valid, error = ActionSchema.validate_action(action)
            if not is_valid:
                return False, f"Action {i}: {error}"
        
        return True, None
    
    def filter_valid_columns(self, target_columns: List[str]) -> Tuple[List[str], List[str]]:
        """Filter target columns to only include those that exist in the DataFrame.
        
        Args:
            target_columns: List of column names to check
            
        Returns:
            Tuple of (valid_columns, invalid_columns)
        """
        valid_columns = [col for col in target_columns if col in self.result_df.columns]
        invalid_columns = [col for col in target_columns if col not in self.result_df.columns]
        return valid_columns, invalid_columns
    
    def execute_action(self, action: Dict[str, Any]) -> bool:
        """Execute a single action.
        
        Args:
            action: Action dictionary following the schema
            
        Returns:
            True if action was executed successfully, False otherwise
        """
        action_type = action.get('type', '').lower()
        target_columns = action.get('target_columns', [])
        conditions = action.get('conditions')
        parameters = action.get('parameters', {})
        row_range = action.get('row_range')  # Optional: {"start": int, "end": int}
        
        # Get action function from registry
        action_func = ACTION_REGISTRY.get(action_type)
        if not action_func:
            st.warning(f"Unknown action type: {action_type}")
            return False
        
        # Filter valid columns
        valid_columns, invalid_columns = self.filter_valid_columns(target_columns)
        
        if invalid_columns:
            st.warning(f"Skipping non-existent columns: {', '.join(invalid_columns)}")
        
        if not valid_columns:
            st.warning(f"Skipping action {action_type}: No valid target columns specified")
            return False
        
        # Validate and normalize row_range
        if row_range:
            max_rows = len(self.result_df)
            start = max(0, min(row_range['start'], max_rows))
            end = max(start, min(row_range['end'], max_rows))
            if start >= max_rows:
                st.warning(f"Row range start ({start}) is beyond DataFrame length ({max_rows})")
                return False
            row_range = {'start': start, 'end': end}
        else:
            # No row_range specified, apply to all rows
            row_range = None
        
        try:
            # Execute the action with valid columns and row_range
            self.result_df = action_func(
                self.result_df,
                valid_columns,
                parameters,
                conditions,
                row_range
            )
            return True
        except Exception as e:
            st.error(f"Error executing {action_type} on columns {valid_columns}: {str(e)}")
            return False
    
    def execute(self, json_data: Dict[str, Any]) -> pd.DataFrame:
        """Execute all actions from JSON data.
        
        Args:
            json_data: JSON data containing actions array
            
        Returns:
            Modified DataFrame after executing all actions
        """
        # Validate schema
        is_valid, error = self.validate_json_schema(json_data)
        if not is_valid:
            st.error(f"Invalid JSON schema: {error}")
            return self.result_df
        
        # Handle empty DataFrame
        if self.result_df.empty:
            num_rows = 10
            if self.result_df.columns.empty:
                st.warning("DataFrame is empty and has no columns")
                return self.result_df
            self.result_df = pd.DataFrame(
                {col: [None] * num_rows for col in self.result_df.columns}
            )
        
        # Execute each action in sequence
        actions = json_data.get('actions', [])
        for i, action in enumerate(actions):
            st.info(f"Executing action {i+1}/{len(actions)}: {action.get('type')}")
            self.execute_action(action)
        
        return self.result_df


def execute_json_actions(df: pd.DataFrame, json_data: Dict[str, Any]) -> pd.DataFrame:
    """Convenience function to execute JSON actions on a DataFrame.
    
    Args:
        df: Input DataFrame
        json_data: JSON data containing actions to execute
        
    Returns:
        Modified DataFrame
    """
    executor = ActionExecutor(df)
    return executor.execute(json_data)
