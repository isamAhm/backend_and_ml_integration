from flask import Flask, request, jsonify
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity
import re

app = Flask(__name__)

# Load pre-trained models and files
tfidf = joblib.load('tfidf.pkl')
svc_model = joblib.load('clf.pkl')
le = joblib.load('encoder.pkl')

# Helper functions
def clean_resume(text):
    # Function to clean the resume text (add your preprocessing here)
    return text.lower()

def extract_years_of_experience(resume_text):
    match = re.search(r'(\d+|\w+)[\s+-]*years', resume_text, re.IGNORECASE)
    if match:
        try:
            years = int(match.group(1)) if match.group(1).isdigit() else text2num(match.group(1))
            return years
        except:
            return 0
    return 0

def text2num(text):
    word_to_num = {
        "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
        "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
    }
    return word_to_num.get(text.lower(), 0)

@app.route('/process-resume', methods=['POST'])
def process_resume():
    try:
        data = request.get_json()
        resumes = data['resumes']
        job_description = data['job_description']
        required_experience = data['required_experience']

        cleaned_job_description = clean_resume(job_description)
        vectorized_job_description = tfidf.transform([cleaned_job_description])

        applicant_rankings = []

        for idx, resume_text in enumerate(resumes):
            cleaned_text = clean_resume(resume_text)
            vectorized_resume = tfidf.transform([cleaned_text])
            predicted_category = svc_model.predict(vectorized_resume.toarray())
            predicted_category_name = le.inverse_transform(predicted_category)[0]

            similarity_score = cosine_similarity(vectorized_resume, vectorized_job_description)[0][0]
            years_of_experience = extract_years_of_experience(resume_text)
            experience_score = min(years_of_experience / required_experience, 1.0)
            final_score = (0.7 * similarity_score) + (0.3 * experience_score)

            applicant_rankings.append({
                "Applicant ID": idx + 1,
                "Resume Text": resume_text,
                "Predicted Category": predicted_category_name,
                "Similarity Score": round(similarity_score, 2),
                "Experience Score": round(experience_score, 2),
                "Final Score": round(final_score, 2)
            })

        sorted_rankings = sorted(applicant_rankings, key=lambda x: x["Final Score"], reverse=True)

        return jsonify({"rankings": sorted_rankings})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
