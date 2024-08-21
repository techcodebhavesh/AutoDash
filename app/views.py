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

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)
    
import os
dirname = os.path.dirname(__file__)

UPLOAD_FOLDER = 'res/'
ALLOWED_EXTENSIONS = {'csv', 'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.json.get('prompt')
    print(prompt)
    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    result = process_prompt(prompt)
    response = result.get("response")
    latest_image_url = result.get("latest_image_url")

    return jsonify({"response": response, "latest_image_url": latest_image_url})


@app.route('/generate_charts', methods=['POST'])
def generate_charts():
    # Extract chart data from the request
    # request_data = request.json
    # if not request_data:
    #     return jsonify({"error": "No data provided"}), 400

    # Assuming 'results' is the chart data generated earlier
    file = ""
    file = uploadFile()
    if file is None:
        file = os.path.join(dirname, "csv/test-new.csv")
        print("File not uploaded. Using the default file: ", file)
    if not file.endswith("test-new.csv"):
        os.remove(file)
        
    file = file.replace("\\", "/")
    return json.dumps(data_to_flask({"file": file}), cls=NpEncoder)


def uploadFile():       
    if 'code' not in request.files:
        return None
    file = request.files['code']
    if file.filename == '':
        return None
    if file:
        _url = os.path.splitext(file.filename)
        filename = secure_filename(_url[0]+"_"+str(uuid.uuid4())+_url[1])
        filename=os.path.join(app.config['UPLOAD_FOLDER'], filename)
        print(filename)
        file.save(filename)
        return filename

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
