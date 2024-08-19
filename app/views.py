from flask import request, jsonify
from app import app
from app.controllers import process_prompt
from app.generateData import data_to_flask
import os
import uuid
from werkzeug.utils import secure_filename
import numpy as np
import json

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
