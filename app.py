import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("AI Resume Matcher - Works 100%")

st.write("Paste your resume text and JD text - no PDF needed!")

resume_text = st.text_area("Paste your RESUME TEXT here", height=200)
jd_text = st.text_area("Paste JOB DESCRIPTION here", height=200)

if st.button("Check Score"):
    if resume_text and jd_text:
        vectorizer = TfidfVectorizer()
        tfidf = vectorizer.fit_transform([resume_text, jd_text])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100
        st.success(f"Match Score: {int(score)}%")
        if score > 70:
            st.balloons()
    else:
        st.warning("Please paste both texts")
