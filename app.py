# app.py
from flask import Flask, request, jsonify
from model_utils import process_text_question, process_audio_question
import os

app = Flask(__name__)

@app.route("/text", methods=["POST"])
def handle_text():
    data = request.get_json()
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "No question provided."}), 400

    response = process_text_question(question)
    return jsonify(response)

@app.route("/audio", methods=["POST"])
def handle_audio():
    if "file" not in request.files:
        return jsonify({"error": "No audio file uploaded."}), 400

    audio_file = request.files["file"]
    audio_path = "uploaded_audio.wav"
    audio_file.save(audio_path)

    response = process_audio_question(audio_path)
    os.remove(audio_path)
    return jsonify(response)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
