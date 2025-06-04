from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import librosa
import soundfile as sf
import os
from transformers import pipeline
from your_model_module import (
    transcribe_audio,
    swer_question,
    save_answer_to_pdf,
    save_conversation_to_docx,
    bookmark_answer,
    save_bookmarks_to_file
)

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Load Whisper ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    tokenizer="openai/whisper-small",
    generate_kwargs={"language": "en"},
    return_timestamps=False
)

@app.route("/text", methods=["POST"])
def handle_text():
    try:
        data = request.get_json()
        question = data.get("question", "")

        if not question:
            return jsonify({"error": "No question provided"}), 400

        question, answer, follow_up, decision = swer_question(question)

        return jsonify({
            "question": question,
            "answer": answer,
            "follow_up": follow_up,
            "decision": decision
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio", methods=["POST"])
def handle_audio():
    try:
        if 'audio' not in request.files:
            return jsonify({"error": "No audio file provided."}), 400

        file = request.files['audio']
        filepath = os.path.join("temp_audio.wav")
        file.save(filepath)

        # Convert to 16kHz mono WAV for Whisper
        audio, sr = librosa.load(filepath, sr=16000)
        sf.write("converted_audio.wav", audio, 16000)

        result = asr_pipeline("converted_audio.wav")
        question = result["text"]

        question, answer, follow_up, decision = swer_question(question)

        return jsonify({
            "transcription": question,
            "answer": answer,
            "follow_up": follow_up,
            "decision": decision
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
