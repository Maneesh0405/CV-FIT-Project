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

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .section-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #667eea;
    }
    
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 15px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-left: 5px solid #28a745;
    }
    
    .selected {
        border-left-color: #28a745;
        background-color: #f8fff9;
    }
    
    .not-selected {
        border-left-color: #dc3545;
        background-color: #fff8f8;
    }
    
    .stButton>button {
        background-color: #667eea !important;
        color: white !important;
        border-radius: 5px !important;
        padding: 10px 20px !important;
        font-weight: bold !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #5a6fd8 !important;
        transform: translateY(-2px) !important;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
    }
    
    .stTextArea textarea, .stFileUploader {
        border-radius: 5px !important;
        border: 1px solid #ced4da !important;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    
    .highlight-text {
        font-weight: bold;
        color: #667eea;
    }
    
    .threshold-info {
        background-color: #e7f3ff;
        border-left: 5px solid #2196F3;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    
    .page-title {
        color: #667eea;
        text-align: center;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Streamlit UI with enhanced styling
st.markdown('<div class="main-header"><h1 style="color: white; margin: 0;">🎯 CV-Fit Tool</h1><p style="color: #e0e0e0; margin: 10px 0 0 0;">Smart Resume Screening & Matching System</p></div>', unsafe_allow_html=True)

# Initialize session state variables
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'job_description'
if 'job_desc' not in st.session_state:
    st.session_state.job_desc = ""
if 'files' not in st.session_state:
    st.session_state.files = []
if 'candidates' not in st.session_state:
    st.session_state.candidates = []
if 'threshold' not in st.session_state:
    st.session_state.threshold = 0.2

# Page routing
if st.session_state.current_page == 'job_description':
    # Job Description Page
    st.markdown('<h2 class="page-title">📋 Job Description</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Enter the job description or upload a PDF file containing the job details.")

    job_desc = st.text_area("Enter Job Description:", height=200, value=st.session_state.job_desc)
    job_file = st.file_uploader("📄 Upload Job Description (PDF)", type=["pdf"], key="job_desc_uploader")

    # Handle file upload
    if job_file is not None:
        extracted_text = extract_text_from_pdf(job_file)
        if extracted_text:
            st.session_state.job_desc = extracted_text
            st.success("✅ Job description extracted from PDF successfully!")
            job_desc = extracted_text

    # Handle text input
    if job_desc and job_desc != st.session_state.job_desc:
        st.session_state.job_desc = job_desc

    # Submit button
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Next ➡️"):
            if st.session_state.job_desc and st.session_state.job_desc.strip():
                st.session_state.current_page = 'upload_resumes'
                st.rerun()
            else:
                st.error("❌ Please enter or upload a job description.")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'upload_resumes':
    # Resume Upload Page
    st.markdown('<h2 class="page-title">👤 Upload Candidate Resumes</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.write("Upload one or more candidate resumes in PDF format for evaluation.")
    
    files = st.file_uploader("📁 Upload Resumes", type=["pdf"], accept_multiple_files=True, key="resume_uploader")
    if files:
        st.session_state.files = files
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.current_page = 'job_description'
            st.rerun()
    with col3:
        if st.button("Next ➡️"):
            if st.session_state.files:
                st.session_state.current_page = 'process_resumes'
                st.rerun()
            else:
                st.error("❌ Please upload at least one resume (PDF).")
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'process_resumes':
    # Processing Page
    st.markdown('<h2 class="page-title">⚙️ Process Resumes</h2>', unsafe_allow_html=True)
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    
    # Threshold Setting
    st.markdown('<div class="threshold-info">', unsafe_allow_html=True)
    st.write("ℹ️ Candidates with a score greater than or equal to the threshold will be selected for the job.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.session_state.threshold = st.slider("🎯 Selection Threshold", min_value=0.0, max_value=1.0, value=st.session_state.threshold, step=0.05)
    
    col1, col2, col3 = st.columns([1, 1, 1])
    with col1:
        if st.button("⬅️ Back"):
            st.session_state.current_page = 'upload_resumes'
            st.rerun()
    with col3:
        if st.button("Process 🚀"):
            if not st.session_state.job_desc:
                st.error("❌ Please enter or upload a job description.")
            elif not st.session_state.files:
                st.error("❌ Please upload at least one resume (PDF).")
            else:
                with st.spinner("🔍 Analyzing resumes... This may take a moment."):
                    candidates = []
                    texts = []
                    for f in st.session_state.files:
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
                        st.error("❌ No valid resumes found.")
                    else:
                        scores = compute_similarity(st.session_state.job_desc, texts, candidates)
                        for i, c in enumerate(candidates):
                            c["score"] = scores[i]
                            # Updated selection logic: candidates with score >= threshold are selected
                            c["status"] = "Selected" if c["score"] >= st.session_state.threshold else "Not Selected"
                        
                        # Store results in session state
                        st.session_state.candidates = candidates
                        st.session_state.current_page = 'results'
                        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

elif st.session_state.current_page == 'results':
    # Results Page
    st.markdown('<h2 class="page-title">📋 Candidate Results</h2>', unsafe_allow_html=True)
    
    candidates = st.session_state.candidates
    
    # Display metrics
    selected_count = sum(1 for c in candidates if c["status"] == "Selected")
    avg_score = sum(c["score"] for c in candidates) / len(candidates)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f'<div class="metric-card"><h2>{len(candidates)}</h2><p>Total Candidates</p></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h2>{selected_count}</h2><p>Selected</p></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h2>{avg_score:.2f}</h2><p>Average Score</p></div>', unsafe_allow_html=True)

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

    # Display threshold information
    st.markdown(f'<div class="threshold-info">Threshold for selection: <strong>{st.session_state.threshold}</strong>. Candidates with scores ≥ {st.session_state.threshold} are marked as <strong>Selected</strong>.</div>', unsafe_allow_html=True)
    
    # Display clean DataFrame (no index, custom numbering)
    st.subheader("📊 Summary Table")
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Download full report
    report_text = "\n\n".join([
        f"No: {i+1}\nName: {c['name']}\nScore: {c['score']:.2f}\nStatus: {c['status']}\nSkills: {', '.join(c['skills'])}\nExperience: {', '.join(c['experience'])}\nProjects: {', '.join(c['projects'])}"
        for i, c in enumerate(candidates)
    ])
    st.download_button("📥 Download Full Candidate Report", report_text, file_name="candidate_report.txt")

    # Back to start button
    if st.button("🔄 Start Over"):
        # Reset session state
        st.session_state.current_page = 'job_description'
        st.session_state.job_desc = ""
        st.session_state.files = []
        st.session_state.candidates = []
        st.session_state.threshold = 0.2
        st.rerun()
    
    st.success("✅ Processing complete! All candidates are displayed with their details.")
