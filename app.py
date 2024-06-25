from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.INFO)


@app.route('/chat', methods=['POST'])
def chat():
    logging.info("Received request: %s", request.json)
    message = request.json.get('message')
    # Simulate a response from the bot
    response = {'reply': f'You said: {message}'}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
