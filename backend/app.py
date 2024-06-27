from flask import Flask, request, jsonify
from flask_cors import CORS
import socket
from translation import translate_mbart
from main import detect_and_set_language, translate_and_respond

app = Flask(__name__)
CORS(app)


@app.route("/chat", methods=["POST"])
def chat():
    if request.method == "POST":
        data = request.get_json()
        if data is None:
            return jsonify({"error": "No data provided in request body"}), 400
        user_input = data.get("message")
        if not user_input:
            return jsonify({"error": "Missing 'message' key in JSON data"}), 400

        # Language detection and translation logic
        detected_lang, source_language_code, detected_language_name = detect_and_set_language(user_input)
        if detected_lang:
            translated_text, translation_time = translate_mbart(user_input, source_language_code,
                                                                target_lang='en_XX', max_length=512, batch_size=8)
            if translated_text:
                response = translate_and_respond(translated_text, source_language_code)
                if 'error' not in response:
                    translated_reply = response.get('translated_reply')
                    return jsonify({"detected_language": detected_language_name, "translated_reply": translated_reply})
                else:
                    error_message = response['error']
                    return jsonify({"detected_language": detected_language_name, "error": error_message})
            else:
                return jsonify({"detected_language": detected_language_name, "error": "Translation failed."})
        else:
            return jsonify({"error": "Language detection failed - Unsupported language."})


if __name__ == "__main__":
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    app.run(host='0.0.0.0', port=8000, debug=True)
