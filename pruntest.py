import pandas as pd
from pandasai import SmartDataframe
from pandasai.llm.local_llm import LocalLLM
import matplotlib.pyplot as plt
# format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format
#magic function for inline charts in notebook
#%matplotlib inline
#For chart style from matplotlib 
plt.style.use('fivethirtyeight')

df = pd.read_csv("student.csv")
df.head()


ollama_llm=LocalLLM(api_base="http://localhost:11434/v1", model="llama3")
df_llm = SmartDataframe(df, config={"llm": ollama_llm})

df_llm.chat('what are the trends of the study time  of students from  studentID 1001 TO 1019 ?')