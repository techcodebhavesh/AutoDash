import pandas as pd
from pandasai import SmartDataframe
from langchain_groq.chat_models import ChatGroq
import os
from firebase_config import bucket

from dotenv import load_dotenv


load_dotenv()

# Format pandas numbers
pd.options.display.float_format = '{:,.0f}'.format

# Load the data
df = pd.read_csv("D:/downloadsD/Data_Analyze_SQL-Packet/csv/test.csv")

# Initialize the LLM and SmartDataframe
llm = ChatGroq(
    model_name="llama3-8b-8192",
    api_key=os.getenv("API_KEY")
)
df_llm = SmartDataframe(df, config={
    "llm": llm,
    "save_charts": True,
    "save_charts_path": "D:/downloadsD/Data_Analyze_SQL-Packet/imgs",
})

watch_directory = "D:/downloadsD/Data_Analyze_SQL-Packet/imgs"
uploaded_images = set()  # Set to keep track of uploaded images

def upload_to_firebase(file_path):
    file_name = os.path.basename(file_path)
    blob = bucket.blob(f"images/{file_name}")
    try:
        print(f"Uploading {file_name} to Firebase Storage...")
        blob.upload_from_filename(file_path)
        blob.make_public()  # Make the blob publicly accessible
        public_url = blob.public_url  # Get the public URL
        print(f"Successfully uploaded {file_name} to Firebase Storage. Public URL: {public_url}")
        return public_url  # Return the public URL
    except Exception as e:
        print(f"Failed to upload {file_name}: {e}")
        return None

# Function to check for new images and upload them
def check_for_new_images(directory):
    print(f"Checking for new images in {directory}...")
    latest_image_url = None
    for file_name in os.listdir(directory):
        if file_name.endswith(('.png', '.jpg', '.jpeg', '.gif')) and file_name not in uploaded_images:
            file_path = os.path.join(directory, file_name)
            print(f"Detected new image: {file_name}. Uploading...")
            latest_image_url = upload_to_firebase(file_path)
            if latest_image_url:
                uploaded_images.add(file_name)  # Add to the set of uploaded images
    return latest_image_url  # Return the URL of the latest uploaded image

# After generating charts, check and upload new images
latest_image_url = check_for_new_images(watch_directory)
