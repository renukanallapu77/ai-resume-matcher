import streamlit as st
import re
import io
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher - ATS Score Checker", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e3a8a, #2563eb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    .score-circle {
        display: flex;
        justify-content: center;
        align-items: center;
        flex-direction: column;
        padding: 1.5rem 0;
    }
    .score-number {
        font-size: 4.5rem;
        font-weight: 800;
        line-height: 1;
    }
    .score-label {
        font-size: 1.2rem;
        color: #64748b;
        margin-top: 0.3rem;
    }
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .skill-tag {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        margin: 0.25rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .skill-have {
        background-color: #dcfce7;
        color: #166534;
        border: 1px solid #86efac;
    }
    .skill-missing {
        background-color: #fee2e2;
        color: #991b1b;
        border: 1px solid #fca5a5;
    }
    .footer {
        text-align: center;
        margin-top: 2.5rem;
        padding-top: 1.2rem;
        border-top: 1px solid #e2e8f0;
        color: #64748b;
        font-size: 0.9rem;
    }
    .footer strong { color: #334155; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">AI Resume Matcher</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">ATS Score Checker — Upload your resume and paste a job description to see how well you match</div>', unsafe_allow_html=True)

COMMON_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "go", "ruby", "php", "swift",
    "kotlin", "rust", "scala", "perl", "r", "matlab", "dart", "html", "css", "react", "angular",
    "vue", "node", "express", "django", "flask", "fastapi", "spring", "hibernate", "dotnet",
    "tensorflow", "pytorch", "keras", "scikit-learn", "pandas", "numpy", "scipy", "matplotlib",
    "seaborn", "tableau", "power bi", "excel", "sql", "mysql", "postgresql", "mongodb",
    "redis", "cassandra", "oracle", "sqlite", "elasticsearch", "graphql", "rest", "soap",
    "docker", "kubernetes", "ansible", "terraform", "jenkins", "aws", "azure", "gcp",
    "google cloud", "linux", "bash", "powershell", "git", "github", "gitlab", "bitbucket",
    "jira", "agile", "scrum", "kanban", "ci/cd", "devops", "machine learning", "deep learning",
    "natural language processing", "nlp", "computer vision", "data science", "data analysis",
    "data engineering", "big data", "hadoop", "spark", "kafka", "airflow", "etl",
    "statistics", "probability", "linear algebra", "calculus", "algorithms", "data structures",
    "object-oriented programming", "oop", "functional programming", "microservices",
    "software development", "unit testing", "pytest", "junit", "selenium", "cypress",
    "html5", "css3", "bootstrap", "tailwind", "jquery", "redux", "webpack", "vite",
    "next.js", "nuxt.js", "svelte", "flutter", "android", "ios", "xcode", "react native",
    "django rest framework", "celery", "rabbitmq", "opencv", "pillow", "beautifulsoup",
    "scrapy", "requests", "flask-restful", "leadership", "communication", "teamwork",
    "problem solving", "critical thinking", "project management", "time management",
    "collaboration", "mentoring", "presentation",
]

def extract_text_from_pdf(file_bytes):
    reader = PdfReader(io.BytesIO(file_bytes))
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def extract_skills(text):
    text_lower = text.lower()
    found = set()
    for skill in COMMON_SKILLS:
        pattern = r"\b" + re.escape(skill) + r"\b"
        if re.search(pattern, text_lower):
            found.add(skill)
    return found

def compute_match_score(resume_text, job_text):
    documents = [resume_text, job_text]
    vectorizer = TfidfVectorizer(stop_words="english", lowercase=True, ngram_range=(1, 2))
    try:
        tfidf_matrix = vectorizer.fit_transform(documents)
    except ValueError:
        return 0.0
    score = cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]
    return float(score * 100)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 📄 Upload Your Resume (PDF)")
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"], label_visibility="collapsed")

with col_right:
    st.markdown("### 📝 Paste Job Description")
    job_description = st.text_area(
        "Job Description",
        height=220,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed",
    )

st.markdown("---")
check_col1, check_col2, check_col3 = st.columns([1, 2, 1])
with check_col2:
    check_clicked = st.button("🔍 Check My Score", use_container_width=True, type="primary")

if check_clicked:
    if not uploaded_file:
        st.warning("Please upload your resume PDF.")
        st.stop()
    if not job_description.strip():
        st.warning("Please paste the job description.")
        st.stop()

    with st.spinner("Analyzing your resume..."):
        resume_text = extract_text_from_pdf(uploaded_file.read())
        if not resume_text.strip():
            st.error("Could not extract any text from the PDF. Please try a different file.")
            st.stop()

        match_score = compute_match_score(resume_text, job_description)
        match_score_rounded = round(match_score)

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)
        skills_have = resume_skills & job_skills
        skills_missing = job_skills - resume_skills

    if match_score_rounded >= 75:
        score_color = "#16a34a"
        verdict = "Great match!"
    elif match_score_rounded >= 50:
        score_color = "#ca8a04"
        verdict = "Decent match — room to improve"
    else:
        score_color = "#dc2626"
        verdict = "Needs significant improvement"

    st.markdown("---")
    st.markdown(
        f"""
        <div class="score-circle">
            <div class="score-number" style="color:{score_color};">{match_score_rounded}% Match</div>
            <div class="score-label">{verdict}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    col_have, col_missing = st.columns(2)

    with col_have:
        st.markdown('<div class="section-header">✅ Skills You Have</div>', unsafe_allow_html=True)
        if skills_have:
            tags = "".join(
                f'<span class="skill-tag skill-have">{s.title()}</span>' for s in sorted(skills_have)
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.info("No directly matching skills detected from the common skills list.")

    with col_missing:
        st.markdown('<div class="section-header">❌ Skills Missing</div>', unsafe_allow_html=True)
        if skills_missing:
            tags = "".join(
                f'<span class="skill-tag skill-missing">{s.title()}</span>' for s in sorted(skills_missing)
            )
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.success("You have all the key skills listed in the job description!")

    st.markdown("---")
    st.markdown(
        """
        <div style="text-align:center; color:#64748b; font-size:0.95rem;">
            Match score calculated using <strong>TF-IDF</strong> and <strong>cosine similarity</strong>
            between your resume and the job description.
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown(
    """
    <div class="footer">
        Built by <strong>Renuka</strong> &nbsp;|&nbsp; B.Tech CSE 3rd Year
    </div>
    """,
    unsafe_allow_html=True,
)
