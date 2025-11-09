# NL-2-Excel: Natural Language to Excel Transformer

An AI-powered Excel file processor that transforms Excel files using natural language instructions. Built with Streamlit and Azure OpenAI.

## Features

- Upload Excel files (.xlsx, .xls)
- Transform data using natural language instructions
- Support for multiple action types: set_value, update, filter, sum, fill_random, delete, derive_column
- Download transformed Excel files

## Prerequisites

- Python 3.8 or higher
- Azure OpenAI account with API access
- Required Python packages (see requirements.txt)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root with your Azure OpenAI credentials:

```bash
cp .env.example .env
```

Then edit `.env` and fill in your Azure OpenAI credentials:

```
AZURE_OPENAI_API_KEY=your_azure_openai_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=your-deployment-name
AZURE_OPENAI_API_VERSION=2024-02-15-preview
```

### 3. Run the Application

```bash
streamlit run excel_processor.py
```

The application will open in your default web browser at `http://localhost:8501`

## Usage

1. **Upload Excel File**: Click "Upload Excel File" and select your .xlsx or .xls file
2. **View Headers**: The application will display all column headers from your file
3. **Enter Instruction**: Type a natural language instruction, for example:
   - "Set all values in column A to 100"
   - "Sum columns B and C into a new column called Total"
   - "Delete column D"
   - "Fill column E with random numbers"
4. **Generate and Execute**: Click "Generate and Execute Plan" to process your instruction
5. **Download Result**: Download the transformed Excel file

## Project Structure

```
.
├── excel_processor.py    # Main Streamlit application
├── actions.py            # Action function definitions
├── executor.py           # JSON parser and action executor
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create from .env.example)
└── README.md            # This file
```

## Action Types

The application supports the following action types:

- **set_value**: Set values in target columns
- **update**: Update columns using formulas
- **filter**: Filter rows based on conditions
- **sum**: Create a sum column from target columns
- **fill_random**: Fill columns with random values
- **delete**: Delete target columns
- **derive_column**: Create a new derived column

## Troubleshooting

### Error: "Error calling OpenAI API"
- Verify your Azure OpenAI credentials in the `.env` file
- Check that your Azure OpenAI deployment is active
- Ensure your API key has the correct permissions

### Error: "Module not found"
- Make sure all dependencies are installed: `pip install -r requirements.txt`

### Error: "Invalid JSON schema"
- The AI-generated JSON might not match the expected schema
- Try rephrasing your instruction
- Check the "View Generated Plan" expander to see what was generated

## License

This project is open source and available for use.
