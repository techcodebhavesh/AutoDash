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
description = df.describe().to_string()

# Prepare the prompt to send to the LLM
prompt = f"""
The following is a description of the DataFrame:
{description}

Please suggest suitable types of graphs for this data and provide Python functions to extract the necessary data for these graphs. Also, integrate the functions into the following template for better understanding and results:

# Template
import pandas as pd

def load_data():
    # Load the CSV file into a DataFrame
    df = pd.read_csv('app/csv/test.csv')
    return df

def plot_graph(data):
    # Add code to plot graphs based on the data
    pass

if __name__ == "__main__":
    df = load_data()
    plot_graph(df)
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

    # Execute the extracted Python code
    exec(python_code)
else:
    print("Python code block not found in the response.")
