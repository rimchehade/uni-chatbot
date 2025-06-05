from flask import Flask, request, jsonify, send_from_directory
from model_utils import handle_text_input, handle_audio_upload
import os
import tempfile

app = Flask(__name__, static_folder='../frontend', static_url_path='')


@app.route('/')
def index():
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)


@app.route('/text', methods=['POST'])
def handle_text():
    data = request.get_json()
    question = data.get("question", "")
    result = handle_text_input(question)

    return jsonify({
        "question": result["question"],
        "answer": result["answer"],
        "follow_up": result["follow_up"],
        "decision": result["decision"]
    })


@app.route('/audio', methods=['POST'])
def handle_audio():
    if 'audio' not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix=".webm") as temp:
        file.save(temp.name)
        result = handle_audio_upload(temp.name)

    return jsonify({
        "question": result["question"],
        "answer": result["answer"],
        "follow_up": result["follow_up"],
        "decision": result["decision"]
    })


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
