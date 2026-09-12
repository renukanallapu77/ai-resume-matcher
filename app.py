import streamlit as st
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Matcher")
st.title("AI Resume Matcher - ATS Score Checker")

resume_file = st.file_uploader("Upload Your Resume PDF", type="pdf")
jd_text = st.text_area("Paste Job Description", height=200)

if st.button("Check My Score"):
    if not resume_file:
        st.error("Please upload resume PDF first")
    elif not jd_text:
        st.error("Please paste Job Description")
    else:
        try:
            reader = PyPDF2.PdfReader(resume_file)
            resume_text = ""
            for page in reader.pages:
                text = page.extract_text()
                if text:
                    resume_text += text

            if not resume_text.strip():
                st.error("Could not read text from PDF. Try a different PDF (not scanned image)")
            else:
                st.info(f"Resume read: {len(resume_text)} characters")

                # Calculate score
                vectorizer = TfidfVectorizer()
                tfidf = vectorizer.fit_transform([resume_text, jd_text])
                score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0] * 100

                st.success(f"## Match Score: {int(score)}%")

                if score > 70:
                    st.balloons()
                    st.markdown("### :green[Great Match! Ready to apply]")
                elif score > 40:
                    st.markdown("### :orange[Medium Match - Add missing skills]")
                else:
                    st.markdown("### :red[Low Match - Update resume]")

        except Exception as e:
            st.error(f"Error: {e}")

st.caption("Built by Renuka | B.Tech CSE 3rd Year")
