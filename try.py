import pandas as pd
from innov8.pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
from firebase_config import bucket

from dotenv import load_dotenv

import os
dirname = os.path.dirname(__file__)
print("*"*50)
print(dirname)
print("*"*50)

load_dotenv()

# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format

# Load the data
df = pd.read_csv(os.path.join(dirname, "D:/AutoDash/app/csv"))

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("API_KEY")
)

df_llm = SmartDataframe(df, config={
    "llm": llm,
    "save_charts": True,
    "save_charts_path": os.path.join(dirname, "..", "imgs"),
})


df_llm.chat("What is the population of the United States?")