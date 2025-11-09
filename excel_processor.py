import os
import json
import streamlit as st
import pandas as pd
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = AzureOpenAI(api_key=os.getenv('AZURE_OPENAI_API_KEY'),
                     azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT'),
                     azure_deployment=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME'),
                     api_version=os.getenv('AZURE_OPENAI_API_VERSION'))

def get_operations_from_ai(headers, instruction):
    """Call OpenAI API to get structured operations based on headers and instruction."""
    try:
        system_prompt = """You are an intelligent assistant that converts natural language Excel manipulation requests into structured JSON instructions.

Your job:
1. Carefully read the available Excel headers: {headers}
2. Interpret the user's intent from their instruction.
3. Even if the user does not use the exact column names, you must choose the most semantically or contextually relevant column name(s) from the provided headers.
4. Always respond ONLY with valid JSON following this schema:

{{
  "actions": [
    {{
      "type": "set_value" | "update" | "filter" | "sum" | "fill_random" | "delete" | "derive_column",
      "target_columns": ["..."],
      "conditions": "optional string if condition exists",
      "parameters": {{
        "value": "optional value",
        "formula": "optional formula or logic"
      }}
    }}
  ]
}}
        """
        
        user_prompt = f"""
        Column Headers: {headers}
        User Instruction: {instruction}
        
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
    st.title("Excel File Processor")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            # Load and process the Excel file
            df = load_excel(uploaded_file)
            
            # Display file info
            st.success(f"Successfully uploaded {uploaded_file.name}")
            display_headers(df)
            
            # User instruction input
            instruction = st.text_area("Enter your instruction or command:")
            
            if st.button("Submit"):
                if not instruction.strip():
                    st.warning("Please enter an instruction before submitting.")
                    return
                    
                with st.spinner("Generating operations..."):
                    # Get column headers
                    headers = df.columns.tolist()
                    
                    # Call OpenAI API
                    operations = get_operations_from_ai(headers, instruction)
                    
                    if operations:
                        st.success("✅ Structured plan generated successfully.")
                        
                        # Display the operations in an expandable section
                        with st.expander("View Generated Operations"):
                            st.json(operations)
                        
                        # Store operations in session state for future use
                        st.session_state['operations'] = operations
                    else:
                        st.error("Failed to generate operations. Please try again.")
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.info("Please upload an Excel file to get started.")

if __name__ == "__main__":
    main()
