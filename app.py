from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords
import os

app = Flask(__name__, static_folder="build", static_url_path="/")
# Allow frontend origin(s) only if you want to restrict; use "*" during testing
CORS(app, resources={r"/*": {"origins": "*"}})

# Make sure stopwords are available
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def clean_text(text):
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


# Serve the frontend index.html (if build/ exists inside backend)
@app.route("/", methods=["GET"])
def serve_home():
    if app.static_folder and os.path.exists(
        os.path.join(app.static_folder, "index.html")
    ):
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({"message": "Backend is running. No frontend build found."})


# POST endpoint for analysis
@app.route("/analyze", methods=["POST"])
def analyze_resume():
    try:
        # Extract form fields
        if "resume" not in request.files or not (
            request.form.get("job_desc") or request.form.get("job_description")
        ):
            return jsonify({"error": "Missing file or job description"}), 400

        resume_file = request.files["resume"]
        # accept either key name from frontend
        job_desc = request.form.get("job_desc") or request.form.get("job_description")

        # Extract and clean text
        resume_text = extract_text_from_pdf(resume_file)
        if not resume_text.strip():
            return jsonify(
                {"error": "Could not extract text from resume (maybe scanned image)."}
            ), 400

        resume_text = clean_text(resume_text)
        job_desc = clean_text(job_desc)

        # TF-IDF vectorization
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        score = round(similarity * 100, 2)

        # Missing keywords
        resume_words = set(resume_text.split())
        job_words = set(job_desc.split())
        missing_keywords = list(job_words - resume_words)

        result = {
            "match_score": score,
            "feedback": f"Your resume matches {score}% with the job description.",
            "missing_keywords": missing_keywords[:15],
        }

        return jsonify(result)

    except Exception as e:
        # Log exception to stdout (Render will capture it)
        print("Error in analyze_resume:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # debug=False for production, Render will run it under gunicorn if configured
    app.run(host="0.0.0.0", port=port, debug=False)
