from flask import request, jsonify
from app import app
from app.controllers import process_prompt
from app.generateData import data_to_flask
import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
import json
from app.ML_MODELS.lr import run as lr_predict
from app.ML_MODELS.pr import run as pr_predict
from app.ML_MODELS.rfr import run as rfr_predict
from app.ML_MODELS.dtr import run as dtr_predict
from app.ML_MODELS.xgbr import run as xgbr_predict
from app.ML_MODELS.catbr import run as catbr_predict
from app.ML_MODELS.lstmm import run as lstmm_run
from app.ML_MODELS.ex import run as ex_run
from app.ML_MODELS.arima import run as arima_run
from app.utility import FileUploader
from app.utility import newestFilePath
import pandas as pd
from datetime import datetime
import openai
from groq import Groq
import pandas as pd

client = Groq(api_key=os.environ.get("API_KEY"))

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # check if obj is a pandas data type
        if str(type(obj)).find("pandas") != -1:
            return obj.to_json(orient='records')
        if isinstance(obj, pd.Timestamp):
            # Convert pandas Timestamp to string
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime):
            # Convert datetime objects to string
            return obj.isoformat()
        # print the type of object to handle it
        print(type(obj))
        return super(NpEncoder, self).default(obj)
    
import os
dirname = os.path.dirname(__file__)

UPLOAD_FOLDER = 'res/'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/ml/test', methods=['GET'])
def test():
    return jsonify({"response": "Hello World!"})


@app.route('/ml/chat', methods=['POST'])
def chat():
    # Check if the content type is multipart/form-data
    if request.content_type.startswith('multipart/form-data'):
        # Handle file upload
        file_uploader = FileUploader(request)
        file = file_uploader.uploadFile()

        if file is None:
            file = os.path.join(dirname, "csv/test-new.csv")
            print("File not uploaded. Using the default file: ", file)
        else:
            print("File uploaded: ", file)
    else:
        # Handle application/json
        file = os.path.join(dirname, "csv/test-new.csv")
        print("Using the default file: ", file)

    # Get the prompt from either form data or JSON
    if request.content_type.startswith('multipart/form-data'):
        prompt = request.form.get('prompt')
    elif request.content_type == 'application/json':
        prompt = request.json.get('prompt')
    else:
        return jsonify({"error": "Unsupported content type"}), 415

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    # Process the prompt
    response, response_type = process_prompt(prompt, file)  # main processing done here
    latest_image_url = newestFilePath(os.environ['NGINX_FOLDER'])

    if response_type == "Error":
        exit()

    print("result")
    print(response)
    print("response_type")
    print(response_type)

    # Delete the uploaded file if necessary
    if request.content_type.startswith('multipart/form-data'):
        file_uploader.deleteFile(file)

    return json.dumps({"response": response, "response_type": response_type, "latest_image_url": latest_image_url}, cls=NpEncoder)


@app.route('/ml/generate_charts', methods=['POST'])
def generate_charts():
    # Extract chart data from the request
    # request_data = request.json
    # if not request_data:
    #     return jsonify({"error": "No data provided"}), 400

    # Assuming 'results' is the chart data generated earlier
    file_uploader = FileUploader(request)
    file = ""
    file = file_uploader.uploadFile()
    if file is None:
        file = os.path.join(dirname, "csv/test-new.csv")
        print("File not uploaded. Using the default file: ", file)
        
    file = file.replace("\\", "/")
    
    output=""
    i=0
    while i<5:
        try:
            output = json.dumps(data_to_flask({"file": file}), cls=NpEncoder)
            break
        except:
            i+=1
            continue
        
    file_uploader.deleteFile(file)

    return output


@app.route('/ml/upload_csv', methods=['POST'])
def upload_csv():
    # Extract chart data from the request
    # request_data = request.json
    # if not request_data:
    #     return jsonify({"error": "No data provided"}), 400

    # Assuming 'results' is the chart data generated earlier
    file_uploader = FileUploader(request)
    file = ""
    file = file_uploader.uploadFile()
    if file is None:
        file = os.path.join(dirname, "csv/test-new.csv")
        print("File not uploaded. Using the default file: ", file)
        
    file = file.replace("\\", "/")
    

    return json.dumps({"filePath":file})


#params
# params.csv_path
# params.target_col
# params.feature_col
# params.user_input
@app.route('/ml/linear_regression', methods=['POST'])
def linear_regression():
    return json.dumps(lr_predict(request.json),cls=NpEncoder)

@app.route('/ml/polynomial_regression', methods=['POST'])
def polynomial_regression():
    return json.dumps(pr_predict(request.json),cls=NpEncoder)

@app.route('/ml/random_forest', methods=['POST'])
def rf_regression():
    return json.dumps(rfr_predict(request.json),cls=NpEncoder)

@app.route('/ml/decision_tree', methods=['POST'])
def dtr_regression():
    return json.dumps(dtr_predict(request.json),cls=NpEncoder)

@app.route('/ml/xg_boost', methods=['POST'])
def xg_boost():
    return json.dumps(xgbr_predict(request.json),cls=NpEncoder)

@app.route('/ml/cat_boost', methods=['POST'])
def cat_boost():
    return json.dumps(catbr_predict(request.json),cls=NpEncoder)

# params
# params.csv_path
# params.date_column
# params.target_column
@app.route('/ml/lstmm', methods=['POST'])
def lstmm():
    return json.dumps(lstmm_run(request.json),cls=NpEncoder)


@app.route('/ml/ex', methods=['POST'])
def ex():
    return json.dumps(ex_run(request.json),cls=NpEncoder)


@app.route('/ml/arima', methods=['POST'])
def arima():
    return json.dumps(arima_run(request.json),cls=NpEncoder)

@app.route('/ml/suggest', methods=['POST'])
def suggest():
    params=request.json
    csv_path = params.get("csv_path")
    df=pd.read_csv(csv_path)

    df_head={df.head(8).to_csv(index=False)}


    

    prompt = f"Suggest a model for predictions from the following options: Linear Regression, Polynomial Regression, Random Forest, Decision Tree, XGBoost, CatBoost, LSTM, Exponential Smoothing, ARIMA. The dataset has the following top 8 rows: {df_head}"

    try:
        # Use the Groq client to create a chat completion
        chat_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            model="llama3-8b-8192",
        )

        # Extract the generated suggestion from the response
        suggestion = chat_completion.choices[0].message.content

        # Return the suggestion in the response
        return jsonify({"suggestion": suggestion})

    except Exception as e:
        # Handle any errors that occur during the API request
        return jsonify({"error": str(e)}), 500
    

