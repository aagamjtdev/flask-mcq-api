import os
import re
import json
import base64
import tempfile
from collections import OrderedDict
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from werkzeug.utils import secure_filename

import fitz  # PyMuPDF
from PIL import Image, ImageEnhance
import pytesseract

# your vector DB helpers
from vector_db import store_mcqs, fetch_mcqs, get_all_mcqs, fetch_random_mcqs

# Google GenAI SDK
import google.generativeai as genai

# --- configuration ---
app = Flask(__name__)
CORS(app)

# Allow overriding tesseract path via env var (not required when we install tesseract in Docker)
tess_cmd = os.getenv("TESSERACT_CMD")
if tess_cmd:
    pytesseract.pytesseract.tesseract_cmd = tess_cmd

# Configure GenAI from environment
GENAI_API_KEY = os.getenv("GENAI_API_KEY")
if not GENAI_API_KEY:
    print("[WARN] GENAI_API_KEY not set — model calls will fail until you set it in Render environment variables.")
else:
    genai.configure(api_key=GENAI_API_KEY)

# create model instance (keeps it at module level so model is reused)
try:
    model = genai.GenerativeModel("gemini-2.0-flash")
except Exception as e:
    model = None
    print("[WARN] Could not initialize GenerativeModel:", e)

# helpers
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

def fix_indic_spacing(text):
    pattern = r'([\u0900-\u097F\u0A80-\u0AFF])\s+([\u0900-\u097F\u0A80-\u0AFF])'
    while re.search(pattern, text):
        text = re.sub(pattern, r'\1\2', text)
    return text

# (keep your get_prompt, extract_text_chunks, format_mcq, extract_mcqs_from_pdf code here)
# For brevity paste the functions you already wrote (I kept them in your earlier message).
# Make sure functions reference `model` and `pytesseract` as above.
# -------------------------
# Minimal endpoints (already in your code)
UPLOAD_FOLDER = "uploaded_pdfs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs("static", exist_ok=True)

@app.route("/extract_mcqs", methods=["POST"])
def upload_mcq():
    user_id = request.form.get("userId")
    title = request.form.get("title")
    description = request.form.get("description")
    pdf_file = request.files.get("pdf")

    if not all([user_id, title, description, pdf_file]):
        return jsonify({"error": "userId, title, description, and pdf file are required"}), 400

    original_name = secure_filename(pdf_file.filename)
    file_name = f"{original_name}"
    file_path = os.path.join(UPLOAD_FOLDER, file_name)
    pdf_file.save(file_path)

    mcqs = extract_mcqs_from_pdf(file_path)
    stored_count = store_mcqs(user_id, title, description, mcqs, file_name)

    return Response(
        json.dumps({
            "generatedQAId": stored_count,
            "userId": user_id,
            "fileName": file_name,
        }, ensure_ascii=False),
        mimetype="application/json"
    )

@app.route("/get_all_mcqs", methods=["GET"])
def api_get_all_mcqs():
    data = get_all_mcqs()
    # safe-format
    for item in data:
        if "mcqs" in item:
            item["mcqs"] = [format_mcq(mcq) for mcq in item["mcqs"]]
    return Response(json.dumps(data, ensure_ascii=False, indent=4), mimetype="application/json")

@app.route("/get_mcqs", methods=["POST"])
def get_mcqs():
    data = request.get_json(silent=True) or request.form.to_dict()
    userId = data.get("userId")
    mcqs_data = fetch_mcqs(user_id=userId)
    if not mcqs_data:
        return jsonify({"message": "No MCQs found"})
    return Response(json.dumps(mcqs_data, ensure_ascii=False, indent=4), mimetype="application/json")

@app.route("/mcq_test", methods=["POST"])
def mcq_test():
    data = request.get_json(silent=True) or request.form
    generatedQAId = data.get("generatedQAId")
    marks = data.get("marks")
    if not generatedQAId:
        return jsonify({"error": "generatedQAId is required"}), 400
    if not marks:
        return jsonify({"error": "marks is required"}), 400
    try:
        marks = int(marks)
    except ValueError:
        return jsonify({"error": "marks must be an integer"}), 400

    mcqs_data = fetch_random_mcqs(generatedQAId, num_questions=marks)
    if not mcqs_data:
        return jsonify({"message": "No MCQs found"}), 404

    return app.response_class(response=mcqs_data, mimetype="application/json"), 200

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=True)
