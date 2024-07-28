import pandas as pd
import os
import json
import re
from langchain_groq.chat_models import ChatGroq
from pandasai import SmartDataframe
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set directory name
dirname = os.path.dirname(__file__)

# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format

# Read CSV file into a DataFrame
df = pd.read_csv(os.path.join(dirname, "app/csv/test.csv"))

# Define dfs as a dictionary of DataFrames
dfs = {
    0: df  # Use df read from the CSV
}

# Initialize LLM
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("API_KEY")
)

# Create SmartDataframe
df_llm = SmartDataframe(df, config={
    "llm": llm,
    "save_charts": True,
    "save_charts_path": os.path.join(dirname, "..", "imgs"),
})

def convert_df_to_csv(df: pd.DataFrame, extras: dict) -> str:
    """
    Convert df to csv-like format where csv is wrapped inside <dataframe></dataframe>
    Args:
        df (pd.DataFrame): PandasAI dataframe or dataframe
        extras (dict): Dictionary that includes 'index'

    Returns:
        str: DataFrame stringify
    """
    dataframe_info = "<dataframe"

    # Add name attribute if available
    if 'name' in extras:
        dataframe_info += f' name="{extras["name"]}"'

    # Add description attribute if available
    if 'description' in extras:
        dataframe_info += f' description="{extras["description"]}"'

    dataframe_info += ">"

    # Add DataFrame details
    dataframe_info += f"\ndfs[{extras['index']}]:{df.shape[0]}x{df.shape[1]}\n{df.head(8).to_csv(index=False)}"

    # Close the DataFrame tag
    dataframe_info += "</dataframe>\n"

    return dataframe_info

# Generate a small description of the DataFrame
description = convert_df_to_csv(df, {"index": 0})

# Define prompts for each chart type
prompts = {
    "bar_chart": f"""
The following is a description of the DataFrame:
{description}

Please provide a complete Python function that:
1. Reads the CSV file from the given path.
2. Extracts the data for the "genre" column.
3. Returns the data in the format: category,value.
Ensure the code is syntactically correct and runnable in Python.
""",
    "pie_chart": f"""
The following is a description of the DataFrame:
{description}

Please provide a complete Python function that:
1. Reads the CSV file from the given path.
2. Extracts data based on the "genre" and "popularity" columns.
3. Returns the data in the format: label,value.
Ensure the code is syntactically correct and runnable in Python.
""",
    "line_chart_single": f"""
The following is a description of the DataFrame:
{description}

Please provide a complete Python function that:
1. Reads the CSV file from the given path.
2. Extracts data for a single line chart based on the "genre" and "duration_ms" columns.
3. Returns the data in the format: date,value.
Ensure the code is syntactically correct and runnable in Python.
""",
    "line_chart_multiple": f"""
The following is a description of the DataFrame:
{description}

Please provide a complete Python function that:
1. Reads the CSV file from the given path.
2. Extracts data for a multiple line chart using the "genre", "popularity", and "duration_ms" columns.
3. Returns the data in the format: date,line1,line2.
Ensure the code is syntactically correct and runnable in Python.
"""
}


def sanitize_quotes(code_str):
    # Replace curly quotes with straight quotes
    return (code_str
            .replace('“', '"')
            .replace('”', '"')
            .replace('‘', "'")
            .replace('’', "'")
            .replace('\u201c', '"')  # Handle specific Unicode characters
            .replace('\u201d', '"'))

def remove_non_printable(code_str):
    # Remove non-printable characters
    return re.sub(r'[^\x00-\x7F]+', '', code_str)

# Function to query LLM and extract code
def get_python_code_for_prompt(prompt):
    response = llm.invoke(prompt)
    response_content = response.content

    # Find the indices of code block delimiters
    start_index = response_content.find("```Python") + 9
    if start_index == 9:  # Adjusting for cases where `Python` keyword might be missing
        start_index = response_content.find("```python") + 9
    if start_index == 9:  # Fallback for generic code blocks
        start_index = response_content.find("```") + 3
    
    if start_index == 3:
        return None
    
    end_index = response_content.find("```", start_index)

    python_code = response_content[start_index:end_index].strip()
    
    # Sanitize the code
    python_code = sanitize_quotes(python_code)
    python_code = remove_non_printable(python_code)
    print(python_code)
    return python_code

# Function to execute code and format output
def execute_and_format_code(code_str, context):
    try:
        # Add dirname and dfs to context
        exec_context = context.copy()
        exec_context['dirname'] = dirname  # Include dirname in the context
        exec_context.update(dfs)  # Include dfs in the context

        # Adjust code to use the correct file path
        code_str = code_str.replace('path_to_your_csv_file.csv', os.path.join(dirname, "app/csv/test.csv"))

        # Execute the user code
        exec(code_str, exec_context)
        
        # Retrieve the result from the execution context
        result = exec_context.get('result', [])
        
        # Return the result and the code used for debugging
        return {
            "code": code_str,
            "result": result
        }
    except Exception as e:
        # Return the code and the error message
        return {
            "code": code_str,
            "error": str(e)
        }

# Prepare the execution context
execution_context = {"pd": pd, "df": df, **dfs}

# Query LLM with each prompt and extract code
extracted_code = {}
for chart_type, prompt in prompts.items():
    print(f"Querying LLM for {chart_type}...")
    code = get_python_code_for_prompt(prompt)
    if code:
        extracted_code[chart_type] = code

# Execute the extracted code and format the output
results = []
for chart_type, code in extracted_code.items():
    print(f"Executing code for {chart_type}...")
    result = execute_and_format_code(code, execution_context)
    
    # Format the results for frontend
    if 'error' in result:
        output = {
            "chartType": chart_type.upper().replace("_", " "),
            "chartData": {
                "error": result['error'],
                "code": result['code']
            }
        }
    else:
        output = {
            "chartType": chart_type.upper().replace("_", " "),
            "chartData": result['result'],
            "code": result['code']  # Include the code used for debugging
        }
    
    results.append(output)

# Print final results
print(json.dumps(results, indent=4))