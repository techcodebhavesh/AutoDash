import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set directory name
dirname = os.path.dirname(__file__)

# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format

# Read CSV file into a DataFrame
df = pd.read_csv(os.path.join(dirname, "app/csv/test.csv"))

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
        extras (dict, optional): expect index to exists

    Returns:
        str: dataframe stringify
    """
    dataframe_info = "<dataframe"

    # Add name attribute if available
    if df.get('name', None) is not None:
        dataframe_info += f' name="{df.name}"'

    # Add description attribute if available
    if df.get('description', None) is not None:
        dataframe_info += f' description="{df.description}"'

    dataframe_info += ">"

    # Add dataframe details
    dataframe_info += f"\ndfs[{extras['index']}]:{df.shape[0]}x{df.shape[1]}\n{df.head(8).to_csv(index=False)}"

    # Close the dataframe tag
    dataframe_info += "</dataframe>\n"

    return dataframe_info

# Generate a small description of the DataFrame
description = convert_df_to_csv(df, {"index": 0})

# Define prompts for each chart type
prompts = {
    "bar_chart": f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable columns and a Python function to extract data for a bar chart, in the format:
category,value
""",
    "pie_chart": f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable columns and a Python function to extract data for a pie chart, in the format:
label,value
""",
    "line_chart_single": f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable columns and a Python function to extract data for a single line chart, in the format:
date,value
""",
    "line_chart_multiple": f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable columns and a Python function to extract data for a multiple line chart, in the format:
date,line1,line2
"""
}

# Function to query LLM and extract code
def get_python_code_for_prompt(prompt):
    response = llm.invoke(prompt)
    response_content = response.content
    
    print(response_content)

    # Check if the response content starts with "python"
    starts_with_python = response_content.strip().startswith("python")

    # Find the indices of code block delimiters
    start_index = response_content.find("```Python") + 9
    if start_index == -1:
        start_index = response_content.find("```python") + 9
    if start_index == -1:
        start_index = response_content.find("```") + 3
    
    end_index = response_content.find("```", start_index)

    python_code = response_content[start_index:end_index].strip()
    
    return python_code


# Query LLM with each prompt and extract code
extracted_code = {}
for chart_type, prompt in prompts.items():
    code = get_python_code_for_prompt(prompt)
    if code:
        extracted_code[chart_type] = code

# Save and optionally execute the extracted code
for chart_type, code in extracted_code.items():
    # Save the code to a file
    filename = f'extracted_code_{chart_type}.py'
    with open(filename, 'w') as file:
        file.write(code)

    # Optionally, execute the code
    print(f"Executing code for {chart_type}...")
    exec(code)
