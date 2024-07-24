import pandas as pd
from pandasai import SmartDataframe   
from langchain_groq.chat_models import ChatGroq
import os


from dotenv import load_dotenv

import os

load_dotenv()
dirname = os.path.dirname(__file__)

# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format
df = pd.read_csv(os.path.join(dirname, "app/csv/test.csv"))

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("API_KEY")
)

df_llm = SmartDataframe(df, config={
    "llm": llm,
    "save_charts": True,
    "save_charts_path": os.path.join(dirname, "..", "imgs"),
})


# df_llm.chat("What is the population of the United States?")

print(llm.invoke("What is the population of the United States?"))

print(df.describe())

from pandasai.helpers.dataframe_serializer import (
    DataframeSerializer,
    DataframeSerializerType,
)

df.name = "Test Dataframe"
df.description = "This is a test dataframe"
df.rows_count = df.shape[0]
df.columns_count = df.shape[1]
df.columns = df.columns.astype(str)


def convert_df_to_csv(df: pd.DataFrame, extras: dict) -> str:
    """
    Convert df to csv like format where csv is wrapped inside <dataframe></dataframe>
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
    dataframe_info += f"\ndfs[{extras['index']}]:{df.rows_count}x{df.columns_count}\n{df.to_csv()}"

    # Close the dataframe tag
    dataframe_info += "</dataframe>\n"

    return dataframe_info

serializer = DataframeSerializerType.CSV
x = convert_df_to_csv(df, {"index": 0})

print(x)