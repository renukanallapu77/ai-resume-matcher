import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher", page_icon="✅")
st.title("✅ AI Resume Matcher - Day 3")
st.write("Paste your Resume and JD below - Get ATS Score")

resume = st.text_area("Paste your RESUME here", height=200)
jd = st.text_area("Paste JOB DESCRIPTION here", height=200)

if st.button("Check Match Score"):
    if resume and jd:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform([resume, jd])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0] * 100
        st.success(f"Your Match Score: {score:.2f}%")
        if score < 60:
            st.warning("Add more keywords from JD to your resume!")
        else:
            st.balloons()
            st.info("Great! You can apply for this job.")
    else:
        st.error("Please paste both Resume and JD")
