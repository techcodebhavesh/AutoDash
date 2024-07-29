from flask import request, jsonify
from app import app
from app.controllers import process_prompt
from app.generateData import data_to_flask

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
    print("helo")
    # Extract chart data from the request
    # request_data = request.json
    # if not request_data:
    #     return jsonify({"error": "No data provided"}), 400

    # Assuming 'results' is the chart data generated earlier
    return jsonify(data_to_flask())