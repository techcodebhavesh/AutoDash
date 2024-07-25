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
    if df.name is not None:
        dataframe_info += f' name="{df.name}"'

    # Add description attribute if available
    if df.description is not None:
        dataframe_info += f' description="{df.description}"'

    dataframe_info += ">"

    # Add dataframe details
    dataframe_info += f"\ndfs[{extras['index']}]:{df.shape[0]}x{df.shape[1]}\n{df.to_csv(index=False)}"

    # Close the dataframe tag
    dataframe_info += "</dataframe>\n"

    return dataframe_info

# Generate a small description of the DataFrame
description = df.describe(include='all').to_string()

# Prepare the prompt to send to the LLM
prompt = f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable types of graphs for this data, and provide Python functions to:
1. Extract the necessary data for each type of graph.
2. Save this data to separate CSV files (e.g., 'bar_chart_data.csv', 'pie_chart_data.csv', 'line_chart_data_single.csv', 'line_chart_data_multiple.csv') in the specified formats.

The formats are as follows:
1. **Bar Chart Format**: 
   - CSV Format: category,value

2. **Pie Chart Format**: 
   - CSV Format: label,value

3. **Line Chart Format (Single Line)**: 
   - CSV Format: date,value

4. **Line Chart Format (Multiple Lines)**: 
   - CSV Format: date,line1,line2

Ensure that the data matches the formats required for each chart type, and the Python functions should handle any necessary data extraction and transformation.

Integrate these functions into the following template for better understanding and results:

# Template
import pandas as pd

def load_data():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('app/csv/test.csv')
    return df

def extract_data_for_bar_chart(data):
    # Define column names
    category_col = 'Category'
    value_col = 'Value'
    
    # Check if columns exist
    if category_col not in data.columns or value_col not in data.columns:
        raise ValueError(f"Columns {{category_col}} and/or {{value_col}} not found in the data.")
    
    # Extract and convert data
    bar_data = {{
        "category": data[category_col].astype(str).tolist(),
        "value": pd.to_numeric(data[value_col], errors='coerce').tolist()  # Convert to numeric, handling errors
    }}
    bar_df = pd.DataFrame(bar_data)
    bar_df.to_csv('bar_chart_data.csv', index=False)

def extract_data_for_pie_chart(data):
    # Define column names
    label_col = 'Label'
    value_col = 'Value'
    
    # Check if columns exist
    if label_col not in data.columns or value_col not in data.columns:
        raise ValueError(f"Columns {{label_col}} and/or {{value_col}} not found in the data.")
    
    # Extract and convert data
    pie_data = {{
        "label": data[label_col].astype(str).tolist(),
        "value": pd.to_numeric(data[value_col], errors='coerce').tolist()  # Convert to numeric, handling errors
    }}
    pie_df = pd.DataFrame(pie_data)
    pie_df.to_csv('pie_chart_data.csv', index=False)

def extract_data_for_line_chart_single(data):
    # Define column names
    date_col = 'Date'
    value_col = 'Value'
    
    # Check if columns exist
    if date_col not in data.columns or value_col not in data.columns:
        raise ValueError(f"Columns {{date_col}} and/or {{value_col}} not found in the data.")
    
    # Extract and convert data
    line_data = {{
        "date": pd.to_datetime(data[date_col], errors='coerce').tolist(),  # Convert to datetime, handling errors
        "value": pd.to_numeric(data[value_col], errors='coerce').tolist()  # Convert to numeric, handling errors
    }}
    line_df = pd.DataFrame(line_data)
    line_df.to_csv('line_chart_data_single.csv', index=False)

def extract_data_for_line_chart_multiple(data):
    # Define column names
    date_col = 'Date'
    line1_col = 'Line1'
    line2_col = 'Line2'
    
    # Check if columns exist
    if date_col not in data.columns or line1_col not in data.columns or line2_col not in data.columns:
        raise ValueError(f"Columns {{date_col}}, {{line1_col}}, and/or {{line2_col}} not found in the data.")
    
    # Extract and convert data
    line_data = {{
        "date": pd.to_datetime(data[date_col], errors='coerce').tolist(),  # Convert to datetime, handling errors
        "line1": pd.to_numeric(data[line1_col], errors='coerce').tolist(),  # Convert to numeric, handling errors
        "line2": pd.to_numeric(data[line2_col], errors='coerce').tolist()   # Convert to numeric, handling errors
    }}
    line_df = pd.DataFrame(line_data)
    line_df.to_csv('line_chart_data_multiple.csv', index=False)

def plot_graphs():
    # Add code to plot graphs based on the extracted data
    pass

if __name__ == "__main__":
    df = load_data()
    extract_data_for_bar_chart(df)
    extract_data_for_pie_chart(df)
    extract_data_for_line_chart_single(df)
    extract_data_for_line_chart_multiple(df)
    plot_graphs()
"""

# Send the prompt to the LLM and get the response
response = llm.invoke(prompt)

# Extract the content from the response
response_content = response.content

# Print the response from the LLM
print(response_content)

# Find the Python code block within the response content
start_index = response_content.find("```")
end_index = response_content.find("```", start_index + 7)

if start_index != -1 and end_index != -1:
    python_code = response_content[start_index + 9:end_index].strip()
    print("Extracted Python Code:")
    print(python_code)

    # Save the extracted Python code to a file
    with open('extracted_code.py', 'w') as file:
        file.write(python_code)
    
    # Execute the extracted Python code
    exec(python_code)
    
else:
    print("Python code block not found in the response.")
