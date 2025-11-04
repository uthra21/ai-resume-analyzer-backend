from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import PyPDF2
import re
import nltk
from nltk.corpus import stopwords

app = Flask(__name__)
CORS(app)

# Make sure stopwords are available
nltk.download("stopwords", quiet=True)
stop_words = set(stopwords.words("english"))


def extract_text_from_pdf(file):
    text = ""
    reader = PyPDF2.PdfReader(file)
    for page in reader.pages:
        text += page.extract_text()
    return text


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    return " ".join(tokens)


@app.route("/analyze", methods=["POST"])
def analyze_resume():
    try:
        # Extract form fields
        if "resume" not in request.files or "job_desc" not in request.form:
            return jsonify({"error": "Missing file or job description"}), 400

        resume_file = request.files["resume"]
        job_desc = request.form["job_desc"]

        # Extract and clean text
        resume_text = extract_text_from_pdf(resume_file)
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
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
