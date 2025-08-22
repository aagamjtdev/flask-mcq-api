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
GENAI_API_KEY = os.getenv("AIzaSyBw4nfwNZPkG3TPT9qLZ2yh2TczBEOu8z0")
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

def get_prompt(text):
    return f"""
You are given raw text or OCR content from a multiple-choice question (MCQ) document.

🔤 The questions and options may be written in English, Hindi, Gujarati, or other Indian languages.

Some questions may include embedded images marked as:
    [IMAGE_QUESTION: filename.png]

Each MCQ may appear in one of the following formats:
- Options labeled as A., B., C., D.
- a) / b) / c)
- 1. / 2. / 3. / 4.
- i. / ii. / iii. / iv.
- Or inside a table
- Or the correct answer appears later as:
  - ANSWER: B
  - Answer - C

Your task is to extract **every complete MCQ** and convert it into valid JSON like this:

[
  {{
    "question": "What is bookkeeping?",
    "image": "page3_img1.png",  # Optional
    "options": {{
      "A": "Records only income",
      "B": "Each entry has two sides",
      "C": "Tracks expenses only",
      "D": "Used for banks only"
    }},
    "answer": "B"
  }}
]

✅ Rules:
- Normalize all option labels
- Remove newlines inside strings
- If a question has [IMAGE_QUESTION: xyz.png], extract the image name and include it in the `"image"` field.
- Return a valid JSON array only
- Each MCQ must have: question text, 2–4 options, and one correct answer
- If text is from OCR and words appear stuck together (like ફસ્ર્લેડીકોનીપત્નીનેકહેવામાંઆવેછે),
  split or normalize them into natural readable words (e.g., "ફર્સ્ટ લેડીની પત્નીને કહેવામાં આવે છે")
Here is the text:

{text}
"""

image_counter = 1
def extract_text_chunks(pdf_path, pages_per_chunk=10, overlap=2):
    """PDF extractor with precise image positioning"""
    global image_counter
    image_counter = 1
    chunks = []
    os.makedirs("static", exist_ok=True)

    doc = fitz.open(pdf_path)
    i = 0

    while i < len(doc):
        chunk_texts = []

        for j in range(i, min(i + pages_per_chunk, len(doc))):
            page = doc[j]
            blocks = []
            is_mcq_page = False

            # 1️⃣ Enhanced text extraction with position tracking
            try:
                page_dict = page.get_text("dict", sort=True)

                for block in page_dict.get("blocks", []):
                    bbox = block.get("bbox", [0, 0, 0, 0])
                    # Use top-left corner for sorting (Y1, X1)
                    sort_key = (bbox[1], bbox[0])

                    if block.get("type") == 0:  # Text block
                        text_block = ""
                        for line in block.get("lines", []):
                            # Preserve natural word spacing
                            for span in line.get("spans", []):
                                text_block += span.get("text", "")
                            text_block += "\n"

                        text_block = text_block.strip()
                        if text_block:
                            # Detect MCQ patterns
                            if re.search(r'^[A-D][\.\)]\s', text_block, re.MULTILINE):
                                is_mcq_page = True
                            blocks.append((sort_key, text_block, "text"))

                    elif block.get("type") == 1:  # Image block
                        try:
                            img_bytes = block.get("image")
                            if img_bytes:
                                img_filename = f"img{image_counter}.png"
                                img_path = os.path.join("static", img_filename)
                                with open(img_path, "wb") as f:
                                    f.write(img_bytes)

                                # Store image with position and type
                                blocks.append((sort_key, f"[IMAGE: {img_filename}]", "image"))
                                image_counter += 1
                        except Exception as e:
                            print(f"[Warning] Failed to save image on page {j + 1}: {e}")

            except Exception as e:
                print(f"[Warning] Failed to parse layout on page {j + 1}: {e}")

            # 2️⃣ OCR fallback wit-h position mapping
            if len(" ".join(t[1] for t in blocks if t[2] == "text").strip()) < 20 or is_mcq_page:
                try:
                    # Create high-resolution image
                    pix = page.get_pixmap(dpi=600)
                    img_filename = f"page_{j + 1}_full.png"
                    img_path = os.path.join("static", img_filename)
                    pix.save(img_path)

                    # Preprocess image
                    img = Image.open(img_path).convert('L')
                    img = ImageEnhance.Contrast(img).enhance(3.0)
                    img = ImageEnhance.Sharpness(img).enhance(3.0)
                    processed_path = os.path.join("static", f"processed_{img_filename}")
                    img.save(processed_path)

                    # OCR with layout preservation
                    custom_config = r"--psm 6 -l eng"
                    if is_mcq_page:
                        custom_config = r"--psm 4 -c preserve_interword_spaces=1"

                    ocr_data = pytesseract.image_to_data(
                        Image.open(processed_path),
                        config=custom_config,
                        output_type=pytesseract.Output.DICT
                    )

                    # Process OCR results with coordinates
                    for i in range(len(ocr_data['text'])):
                        text = ocr_data['text'][i]
                        if text.strip():
                            x = ocr_data['left'][i]
                            y = ocr_data['top'][i]
                            blocks.append(((y, x), text, "ocr"))

                    os.remove(processed_path)
                except Exception as e:
                    print(f"[Error] OCR failed on page {j + 1}: {e}")

            # 3️⃣ Precise sorting using coordinates
            try:
                # Sort blocks by Y then X coordinates
                blocks.sort(key=lambda x: (x[0][0], x[0][1]))

                # Assemble page text
                page_text = []
                current_line_y = -1
                current_line = []

                for pos, text, block_type in blocks:
                    y = pos[0]

                    # Group elements on the same line
                    if current_line_y < 0 or abs(y - current_line_y) < 15:  # 15px tolerance
                        current_line.append(text)
                        current_line_y = y
                    else:
                        # Start new line
                        page_text.append(" ".join(current_line))
                        current_line = [text]
                        current_line_y = y

                if current_line:
                    page_text.append(" ".join(current_line))

                page_text = "\n".join(page_text)

                # Fix MCQ-specific formatting
                if is_mcq_page:
                    # Fix spaced-out letters (B. I W A C → B. IWAC)
                    page_text = re.sub(r'(\b[A-D]) (?=\w)', r'\1', page_text)
                    # Fix line breaks in options
                    page_text = re.sub(r'([A-D])\.\s*\n\s*', r'\1. ', page_text)

            except Exception as e:
                print(f"Sorting failed: {e}")
                page_text = "\n".join(t[1] for t in blocks)

            if page_text.strip():
                chunk_texts.append(f"=== PAGE {j + 1} ===\n{page_text}")

        if chunk_texts:
            chunks.append("\n\n".join(chunk_texts))

        i += pages_per_chunk - overlap

    doc.close()
    return chunks

def format_mcq(mcq):
    return {
        "question": mcq.get("question") or mcq.get("ques") or mcq.get("q"),
        "image": mcq.get("image") or mcq.get("img"),
        "options": mcq.get("options") or mcq.get("opts"),
        "answer": mcq.get("answer") or mcq.get("ans") or mcq.get("correct")
    }

def extract_mcqs_from_pdf(pdf_path):
    chunks = extract_text_chunks(pdf_path)
    all_mcqs = []
    seen = set()

    for idx, chunk in enumerate(chunks):
        prompt = get_prompt(chunk)
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip()
            print(f"[Gemini chunk {idx+1}]:", raw)

            m = re.search(r"\[\s*{.*?}\s*\]", raw, re.DOTALL)
            if not m:
                print(f"[Skip] No valid JSON in chunk {idx+1}")
                continue

            arr = json.loads(m.group())
            if not isinstance(arr, list):
                continue
            """
            #image to base64
            for mcq in arr:
                q = mcq.get("question", "").strip()
                opts = mcq.get("options", {})
                ans = mcq.get("answer", "").strip()
                img_name = mcq.get("image", "")

                key = q.lower()
                if q and opts and ans and key not in seen:
                    seen.add(key)
                    base64_img = encode_image_to_base64(os.path.join("static", img_name)) if img_name else None
                    all_mcqs.append(OrderedDict([
                        ("question", q),
                        ("options", opts),
                        ("answer", ans),
                        ("image_base64", base64_img)
                    ]))
            """
            for mcq in arr:
                q = mcq.get("question", "").strip()
                opts = mcq.get("options", {})
                ans = mcq.get("answer", "").strip()
                img = mcq.get("image", None)

                key = q.lower()
                if q and opts and ans and key not in seen:
                    seen.add(key)
                    all_mcqs.append(OrderedDict([
                        ("question", q),
                        ("image", img),
                        ("options", opts),
                        ("answer", ans)
                    ]))

        except Exception as e:
            print(f"❌ Error chunk {idx+1}:", e)
            continue

    return all_mcqs

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
