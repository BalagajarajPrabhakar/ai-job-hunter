import requests
import smtplib
import json
import os
from email.mime.text import MIMEText
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import sent_tokenize

nltk.download('punkt')

# ========= CONFIG ==========
EMAIL_ADDRESS = os.environ.get("EMAIL_ADDRESS")
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")
TO_EMAIL = os.environ.get("TO_EMAIL")

KEYWORDS = ["python", "developer", "backend", "software"]
LOCATION_FILTER = ""        # change as needed
MIN_SALARY =                # yearly USD
MAX_JOBS = 10
# ============================

def fetch_jobs():
    url = "https://remotive.com/api/remote-jobs"
    response = requests.get(url)
    return response.json()["jobs"]

def load_saved_jobs():
    if os.path.exists("saved_jobs.json"):
        with open("saved_jobs.json", "r") as f:
            return json.load(f)
    return []

def save_jobs(job_ids):
    with open("saved_jobs.json", "w") as f:
        json.dump(job_ids, f)

def filter_jobs(jobs, saved_ids):
    filtered = []
    for job in jobs:
        title = job["title"].lower()
        location = job["candidate_required_location"].lower()
        salary = job.get("salary", "")
        job_id = job["id"]

        if job_id in saved_ids:
            continue

        if not any(k in title for k in KEYWORDS):
            continue

        if LOCATION_FILTER not in location:
            continue

        if salary:
            numbers = [int(s) for s in salary.split() if s.isdigit()]
            if numbers and max(numbers) < MIN_SALARY:
                continue

        filtered.append(job)

        if len(filtered) >= MAX_JOBS:
            break

    return filtered

def summarize(text):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:3])

def score_job(cv_text, job_text):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform([cv_text, job_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)

def format_email(jobs, cv_text):
    message = f"🚀 Daily AI Job Hunter - {datetime.now().strftime('%Y-%m-%d')}\n\n"

    for job in jobs:
        summary = summarize(job["description"])
        match_score = score_job(cv_text, job["description"])

        message += f"🔹 {job['title']} ({match_score}% match)\n"
        message += f"Company: {job['company_name']}\n"
        message += f"Location: {job['candidate_required_location']}\n"
        message += f"Apply: {job['url']}\n"
        message += f"Summary: {summary}\n\n"

    return message

def send_email(content):
    msg = MIMEText(content)
    msg["Subject"] = "🚀 Daily AI Job Hunter"
    msg["From"] = EMAIL_ADDRESS
    msg["To"] = TO_EMAIL

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.send_message(msg)

def main():
    jobs = fetch_jobs()
    saved_ids = load_saved_jobs()

    with open("my_cv.txt", "r") as f:
        cv_text = f.read()

    new_jobs = filter_jobs(jobs, saved_ids)

    if new_jobs:
        email_content = format_email(new_jobs, cv_text)
        send_email(email_content)

        updated_ids = saved_ids + [job["id"] for job in new_jobs]
        save_jobs(updated_ids)

        print("AI Job Hunter ran successfully.")
    else:
        print("No new matching jobs today.")

if __name__ == "__main__":
    main()
