import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.title("📄 AI Resume Ranker")
st.subheader("Match your Resume with a Job Description")

job_desc = st.text_area("🔍 Job Description", height=200)
resume = st.text_area("🧑‍💼 Resume Text", height=200)

if st.button("💡 Rank Resume"):
    if job_desc and resume:
        tfidf = TfidfVectorizer()
        vectors = tfidf.fit_transform([job_desc, resume])
        score = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
        st.success(f"🎯 Match Score: {round(score * 100, 2)}%")

        if score > 0.75:
            st.info("✅ Excellent Fit")
        elif score > 0.5:
            st.warning("⚠️ Average Fit – Improve Keywords")
        else:
            st.error("❌ Not a Good Match")
    else:
        st.error("Please fill both fields!")
