from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("API_KEY")
)


def convert_df_to_csv(df: pd.DataFrame, extras: dict) -> str:
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

def process_prompt(prompt, filepath):
    i = 0
    while True:
        i+=1
        try:
            df = pd.read_csv(filepath)
            df_llm = SmartDataframe(df, config={
                "llm": llm,
                "save_charts": True,
                "save_charts_path": os.path.join(os.environ['NGINX_FOLDER']),
            })
            response = df_llm.chat(prompt)
            print("RESP   ")
            print(response)
            
            
            if isinstance(response, dict):
                if 'data' in response and isinstance(response['data'], dict) and 'image' in response['data']:
                    response_type = "Plot"
                elif 'figure' in response and response['figure'] is not None:
                    response_type = "Plot"
                else:
                    response_type = "Unknown Type"
                    
            if os.path.isfile(response):
                response_type = "Plot"
            elif isinstance(response, pd.DataFrame):
                response_type = "DataFrame"
            elif isinstance(response, (int, float)):
                response_type = "Number"
            elif isinstance(response, str):
                response_type = "String"
            else:
                response_type = "Unknown Type"
                
            
            
            # After generating charts, check and upload new images
            # from app.models import check_for_new_images, watch_directory
            # print("Checking for new images to upload after generating charts...")
            # latest_image_url = check_for_new_images(watch_directory)
            
            if not response.startswith("Unfortunately"):
                return response, response_type
            
            if i > 5:
                return f"Couldn't curate a valid answer for you. Please Try changing the prompt. Try being exact and precise. Here is the sample header of the data for you reference \n{convert_df_to_csv(df)}", "Error"
        except Exception as e:
            return str(e), "Error"
