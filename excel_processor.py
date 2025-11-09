import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from executor import execute_json_actions

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AzureOpenAI(api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                     azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                     api_version=os.getenv('AZURE_OPENAI_API_VERSION'))

def get_operations_from_ai(headers, instruction, num_rows=None):
    """Call OpenAI API to get structured operations based on headers and instruction.
    
    Args:
        headers: List of column headers
        instruction: User's natural language instruction
        num_rows: Optional number of rows in the dataset (for "last N rows" calculations)
    """
    try:
        system_prompt = """You are an intelligent assistant that converts natural language Excel manipulation requests into structured JSON instructions.

Your job:
1. Carefully read the available Excel headers: {headers}
2. Interpret the user's intent from their instruction.
3. Even if the user does not use the exact column names, you must choose the most semantically or contextually relevant column name(s) from the provided headers.
4. Determine which rows to manipulate based on the instruction (e.g., "first 10 rows", "rows 5-15", "last 5 rows").
5. Always respond ONLY with valid JSON following this schema:

{{
  "actions": [
    {{
      "type": "set_value" | "update" | "filter" | "sum" | "fill_random" | "delete" | "derive_column",
      "target_columns": ["..."],
      "row_range": {{
        "start": 0,
        "end": 10
      }},
      "conditions": "optional string if condition exists",
      "parameters": {{
        "value": "optional value",
        "formula": "optional formula or logic",
        "new_column": "optional new column name for sum/derive_column",
        "min": "optional minimum value for numeric ranges (fill_random)",
        "max": "optional maximum value for numeric ranges (fill_random)",
        "data_type": "optional: 'numeric', 'integer', 'categorical', 'boolean', 'string', 'date'",
        "categories": "optional array of values for categorical data (e.g., [\"yes\", \"no\"], [\"True\", \"False\"])",
        "integer_only": "optional boolean for numeric data (true for integers, false for floats)"
      }}
    }}
  ]
}}

Row Range Rules:
- "row_range" is optional. If not specified, the action applies to ALL rows.
- "row_range.start": inclusive starting row index (0-based, like Python)
- "row_range.end": exclusive ending row index (like Python slicing, so end=10 means rows 0-9)
- If only specific rows are mentioned, use row_range with appropriate start and end values.
- If "all rows" or no row specification, omit row_range entirely.

Fill Random Action Rules:
- For numeric columns (Age, Price, Amount, etc.): Use "min" and "max" parameters for range (e.g., min: 1, max: 100)
- For categorical columns (Status, Category, etc.): Use "categories" parameter with array of possible values
  - Common patterns: ["yes", "no"], ["True", "False"], ["Active", "Inactive"], ["High", "Medium", "Low"]
- For boolean columns: Use "data_type": "boolean" or "categories": ["True", "False"]
- For integer-only numeric: Set "integer_only": true and use "min"/"max"
- Infer data type from column name context:
  - Columns with "status", "active", "enabled" -> categorical/boolean
  - Columns with "age", "price", "amount", "count", "number" -> numeric
  - Columns with "date", "time" -> date
  - Columns with "name", "description", "text" -> string
- If user specifies a range (e.g., "1-100"), set min: 1, max: 100
- If user specifies categories (e.g., "yes/no"), set categories: ["yes", "no"]
        """
        
        row_info = f"\nTotal number of rows in the dataset: {num_rows}" if num_rows is not None else ""
        user_prompt = f"""
        Column Headers: {headers}{row_info}
        User Instruction: {instruction}
        
        Important: Determine which rows to manipulate from the instruction. Examples:
        - "first 10 rows" -> row_range: {{"start": 0, "end": 10}}
        - "rows 5 to 15" -> row_range: {{"start": 5, "end": 16}} (end is exclusive, so this covers rows 5-15)
        - "last 5 rows" -> If total rows is 100, then row_range: {{"start": 95, "end": 100}}
        - "all rows" or no row specification -> omit row_range entirely
        
        For fill_random actions, analyze the column name and user instruction to determine:
        - Data type (numeric, categorical, boolean, string, date)
        - Range (min/max for numeric) or categories (array for categorical)
        - Examples:
          * "fill random values in range 1-100" -> min: 1, max: 100, data_type: "numeric"
          * "fill random yes/no values" -> categories: ["yes", "no"], data_type: "categorical"
          * "fill random status" (for Status column) -> categories: ["Active", "Inactive"] or ["Yes", "No"]
          * "fill random integers from 10 to 50" -> min: 10, max: 50, integer_only: true
        
        Return only the JSON response with the operations.
        """
        
        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),  # or "gpt-4o" when available
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3
        )
        
        # Parse the response to get clean JSON
        content = response.choices[0].message.content
        return json.loads(content)
    except Exception as e:
        st.error(f"Error calling OpenAI API: {str(e)}")
        return None

def load_excel(uploaded_file):
    """Load Excel file into a pandas DataFrame and save temporarily."""
    df = pd.read_excel(uploaded_file)
    # Save the file temporarily
    temp_path = Path("uploaded.xlsx")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return df

def display_headers(df):
    """Display column headers in a formatted way."""
    st.write("### Column Headers:")
    st.write(df.columns.tolist())


def main():
    st.title("Excel File Processor with AI-Powered Transformations")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Load and process the Excel file
            df = load_excel(uploaded_file)
            
            # Display file info
            st.success(f"Successfully uploaded {uploaded_file.name}")
            
            # Show data preview
            st.write("### Data Preview")
            display_headers(df)
            
            # User instruction input
            instruction = st.text_area("Enter your instruction or command (e.g., 'Set all values in column A to 100'):")
            
            if st.button("Generate and Execute Plan"):
                if not instruction.strip():
                    st.warning("Please enter an instruction before submitting.")
                    return
                    
                with st.spinner("Generating and executing plan..."):
                    # Get column headers and row count
                    headers = df.columns.tolist()
                    num_rows = len(df)
                    
                    # Call OpenAI API
                    operations = get_operations_from_ai(headers, instruction, num_rows)
                    
                    if operations:
                        st.success("✅ Plan generated successfully!")
                        
                        # Display the operations
                        with st.expander("View Generated Plan"):
                            st.json(operations)
                        
                        # Execute the operations
                        result_df = execute_json_actions(df, operations)
                        
                        # Show the result
                        st.write("### Transformed Data")
                        st.dataframe(result_df)
                        
                        # Save to Excel
                        output_path = "output.xlsx"
                        result_df.to_excel(output_path, index=False)
                        
                        # Provide download button
                        with open(output_path, "rb") as file:
                            st.download_button(
                                label="Download Transformed Excel",
                                data=file,
                                file_name="transformed_output.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                            
                    else:
                        st.error("Failed to generate operations. Please try again.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.exception(e)  # Show detailed error in debug mode
    else:
        st.info("Please upload an Excel file to get started.")

if __name__ == "__main__":
    main()
