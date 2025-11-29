import streamlit as st
import sqlite3
from io import BytesIO
from PyPDF2 import PdfReader
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import pandas as pd

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    st.error("spaCy model not found. Run: python -m spacy download en_core_web_sm")
    st.stop()

# SQLite setup
conn = sqlite3.connect('cv_fit.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS jobs (id INTEGER PRIMARY KEY, description TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS candidates (id INTEGER PRIMARY KEY, name TEXT, resume_text TEXT, skills TEXT, experience TEXT)''')
conn.commit()

# Function to extract text from PDF
def extract_text_from_pdf(file):
    try:
        reader = PdfReader(BytesIO(file.read()))
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        text = re.sub(r'\s+', ' ', text).strip()
        if not text:
            raise ValueError("No text extracted.")
        return text
    except Exception as e:
        st.error(f"Text extraction failed for {file.name}: {str(e)}")
        return ""

# Function to extract skills, experience, and projects
def extract_entities(text):
    doc = nlp(text.lower())
    skill_keywords = [
        "python", "sql", "machine learning", "ml", "data analysis", "data analytics",
        "javascript", "js", "java", "c++", "html", "css", "deep learning"
    ]
    skills = set()
    for token in doc:
        if token.text in skill_keywords:
            skills.add(token.text.title())

    years_matches = re.findall(r'\b\d+\s*years?\b', text, re.IGNORECASE)
    job_titles = re.findall(r'\b(?:engineer|analyst|developer|manager|specialist|intern|consultant)\b', text, re.IGNORECASE)
    role_matches = re.findall(r'(?:worked as|experience in|role as)\s*[\w\s]+', text, re.IGNORECASE)
    experience = list(set(years_matches + job_titles + role_matches))

    project_matches = re.findall(r'(?:project|built|developed|created|learned)\s*[\w\s,.-]+', text, re.IGNORECASE)
    projects = list(set(project_matches))

    return list(skills), experience, projects

# Compute similarity
def compute_similarity(job_desc, resumes, candidates):
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        texts = [job_desc.lower()] + [r.lower() for r in resumes]
        tfidf_matrix = vectorizer.fit_transform(texts)
        tfidf_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

        job_skills = extract_entities(job_desc)[0]
        skill_scores = []
        for cand in candidates:
            matched = len(set(cand['skills']) & set(job_skills))
            total = len(job_skills) or 1
            skill_scores.append(matched / total)

        combined = [0.7 * t + 0.3 * s for t, s in zip(tfidf_scores, skill_scores)]
        return combined
    except Exception as e:
        st.error(f"Similarity error: {str(e)}")
        return [0.0] * len(resumes)

# Streamlit UI
st.title("CV-Fit Tool")

st.header("1. Job Description")
job_desc = st.text_area("Enter Job Description (Or Upload PDF Below):")

job_file = st.file_uploader("Upload Job Description (PDF)", type=["pdf"])
if job_file:
    job_desc = extract_text_from_pdf(job_file)

st.header("2. Upload Candidate Resumes (PDF)")
files = st.file_uploader("Upload Resumes", type=["pdf"], accept_multiple_files=True)

threshold = 0.2

if st.button("Process"):
    if not job_desc:
        st.error("Please enter or upload a job description.")
    elif not files:
        st.error("Please upload at least one resume (PDF).")
    else:
        candidates = []
        texts = []
        for f in files:
            txt = extract_text_from_pdf(f)
            if txt:
                sk, exp, proj = extract_entities(txt)
                name = f.name.split('.')[0]
                candidates.append({
                    "name": name,
                    "skills": sk,
                    "experience": exp,
                    "projects": proj,
                    "score": 0.0
                })
                texts.append(txt)

        if not candidates:
            st.error("No valid resumes found.")
        else:
            scores = compute_similarity(job_desc, texts, candidates)
            for i, c in enumerate(candidates):
                c["score"] = scores[i]
                c["status"] = "Selected" if c["score"] >= threshold else "Not Selected"

            # Create DataFrame with numbering starting from 1
            df = pd.DataFrame([
                {
                    "No.": i + 1,
                    "Name": c["name"],
                    "Score": round(c["score"], 2),
                    "Status": c["status"],
                    "Skills": ", ".join(c["skills"]) if c["skills"] else "None",
                    "Experience": ", ".join(c["experience"]) if c["experience"] else "None",
                    "Projects": ", ".join(c["projects"]) if c["projects"] else "None"
                }
                for i, c in enumerate(candidates)
            ])

            st.header("Candidate Results")

            # Display clean DataFrame (no index, custom numbering)
            st.dataframe(df, use_container_width=True, hide_index=True)

            # Download full report
            report_text = "\n\n".join([
                f"No: {i+1}\nName: {c['name']}\nScore: {c['score']:.2f}\nStatus: {c['status']}\nSkills: {', '.join(c['skills'])}\nExperience: {', '.join(c['experience'])}\nProjects: {', '.join(c['projects'])}"
                for i, c in enumerate(candidates)
            ])
            st.download_button("Download Full Candidate Report", report_text)

            st.success("Processing complete! All candidates are displayed with their details.")
