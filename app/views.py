from flask import request, jsonify
from app import app
from app.controllers import process_prompt

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
