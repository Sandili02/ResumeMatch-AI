
import re
import io
import fitz
import pandas as pd
import streamlit as st
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ResumeMatch AI",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
# 2. SKILL TAXONOMY AND ALIASES
# --------------------------------------------------

SKILL_TAXONOMY = {
    "Programming": [
        "Python", "SQL", "R Programming", "Java", "C++", "C#"
    ],
    "Data Analysis": [
        "Excel", "Power BI", "Tableau", "Pandas", "NumPy",
        "Matplotlib", "Seaborn", "Statistics",
        "Exploratory Data Analysis", "Data Visualization",
        "Data Cleaning"
    ],
    "Machine Learning & AI": [
        "Machine Learning", "Deep Learning",
        "Natural Language Processing", "TensorFlow",
        "PyTorch", "Scikit-learn"
    ],
    "Cloud & Tools": ["AWS", "Azure", "Git"],
    "Professional Skills": ["Communication", "Problem Solving"]
}

ALIASES = {
    "Power BI": ["powerbi", "power-bi", "microsoft power bi"],
    "SQL": ["structured query language", "sql queries"],
    "Machine Learning": ["ml", "machine-learning"],
    "Natural Language Processing": ["nlp"],
    "Exploratory Data Analysis": ["eda"],
    "Scikit-learn": ["sklearn", "scikit learn"],
    "AWS": ["amazon web services"],
    "Azure": ["microsoft azure"],
    "Excel": ["ms excel", "microsoft excel"]
}

def get_skill_terms():
    terms = {}

    for skills in SKILL_TAXONOMY.values():
        for skill in skills:
            terms[skill] = [skill]

    for canonical, aliases in ALIASES.items():
        terms.setdefault(canonical, [canonical])
        terms[canonical].extend(aliases)

    return terms

SKILL_TERMS = get_skill_terms()

# --------------------------------------------------
# 3. PDF TEXT EXTRACTION
# --------------------------------------------------

def extract_pdf_text(pdf_bytes):
    if not pdf_bytes:
        raise ValueError("The uploaded PDF is empty.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("Please upload a valid PDF resume.")

    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            if doc.is_encrypted:
                raise ValueError(
                    "This PDF is password-protected. Upload an unlocked PDF."
                )

            if len(doc) == 0:
                raise ValueError("The PDF contains no pages.")

            if len(doc) > 30:
                raise ValueError(
                    "Please upload a resume containing 30 pages or fewer."
                )

            pages = [
                page.get_text("text", sort=True).strip()
                for page in doc
            ]

    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Unable to read the PDF: {exc}") from exc

    text = "\n\n".join(pages)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    if len(text) < 50:
        raise ValueError(
            "Very little text was extracted. This may be a scanned PDF "
            "and may require OCR."
        )

    return text, len(pages)

# --------------------------------------------------
# 4. SKILL EXTRACTION
# --------------------------------------------------

def extract_skills(text):
    found = set()
    text_lower = text.lower()

    for canonical, variants in SKILL_TERMS.items():
        for variant in set(variants):
            pattern = (
                r"(?<!\w)" + re.escape(variant.lower()) + r"(?!\w)"
            )

            if re.search(pattern, text_lower):
                found.add(canonical)
                break

    return found

# --------------------------------------------------
# 5. TEXT SIMILARITY
# --------------------------------------------------

def calculate_similarity(resume_text, job_text):
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True
    )

    matrix = vectorizer.fit_transform([
        resume_text,
        job_text
    ])

    return float(
        cosine_similarity(matrix[0:1], matrix[1:2])[0, 0]
    ) * 100

# --------------------------------------------------
# 6. APPLICATION HEADER
# --------------------------------------------------

st.title("🎯 ResumeMatch AI")
st.subheader("Resume–Job Description Matching System")

st.write(
    "Compare your resume with a job description, discover skill overlaps, "
    "and identify potential areas for improvement."
)

st.info(
    "This tool supports candidates in reviewing their resumes. "
    "Its scores are informational indicators, not hiring predictions."
)

# --------------------------------------------------
# 7. INPUT SECTION
# --------------------------------------------------

left, right = st.columns(2)

with left:
    uploaded_file = st.file_uploader(
        "📄 Upload Resume (PDF)",
        type=["pdf"]
    )

with right:
    job_description = st.text_area(
        "💼 Paste Job Description",
        height=220,
        placeholder="Paste the complete job description here..."
    )

analyze = st.button(
    "🚀 Analyze Resume Match",
    type="primary",
    use_container_width=True
)

# --------------------------------------------------
# 8. ANALYSIS ENGINE
# --------------------------------------------------

if analyze:

    if uploaded_file is None:
        st.error("Please upload a PDF resume.")
        st.stop()

    if len(job_description.strip()) < 50:
        st.error("Please enter a job description of at least 50 characters.")
        st.stop()

    if len(job_description.strip()) > 20000:
        st.error("Please limit the job description to 20,000 characters.")
        st.stop()

    try:
        resume_text, page_count = extract_pdf_text(
            uploaded_file.getvalue()
        )
    except ValueError as exc:
        st.error(str(exc))
        st.stop()

    with st.spinner("Analyzing resume and job description..."):

        resume_skills = extract_skills(resume_text)
        job_skills = extract_skills(job_description)

        matched_skills = sorted(
            resume_skills.intersection(job_skills)
        )

        missing_skills = sorted(
            job_skills.difference(resume_skills)
        )

        additional_skills = sorted(
            resume_skills.difference(job_skills)
        )

        text_similarity = calculate_similarity(
            resume_text, job_description
        )

        if job_skills:
            skill_coverage = (
                len(matched_skills) / len(job_skills)
            ) * 100
        else:
            skill_coverage = 0.0

        overall_score = (
            0.50 * text_similarity
            + 0.50 * skill_coverage
        )

    # --------------------------------------------------
    # 9. MATCHING METRICS
    # --------------------------------------------------

    st.divider()
    st.header("📊 Matching Results")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Overall Matching Indicator",
        f"{overall_score:.1f}%"
    )

    c2.metric(
        "Text Similarity",
        f"{text_similarity:.1f}%"
    )

    c3.metric(
        "Skill Coverage",
        f"{skill_coverage:.1f}%"
    )

    st.caption(
        "Overall indicator = 50% text similarity + 50% skill coverage. "
        "This configurable formula is not a validated hiring score."
    )

    st.success(
        f"Resume processed successfully: {page_count} PDF page(s)."
    )

    # --------------------------------------------------
    # 10. SKILL TABLES
    # --------------------------------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Dashboard",
        "✅ Matched Skills",
        "🔍 Skill Gaps",
        "📄 Extracted Resume"
    ])

    with tab1:

        score_df = pd.DataFrame({
            "Metric": [
                "Text Similarity",
                "Skill Coverage",
                "Overall Indicator"
            ],
            "Percentage": [
                text_similarity,
                skill_coverage,
                overall_score
            ]
        })

        fig = px.bar(
            score_df,
            x="Metric",
            y="Percentage",
            color="Metric",
            text_auto=".1f",
            range_y=[0, 100],
            title="Resume Matching Overview"
        )

        st.plotly_chart(fig, use_container_width=True)

        category_rows = []

        for category, skills in SKILL_TAXONOMY.items():

            required = set(skills).intersection(job_skills)
            matched = required.intersection(resume_skills)

            if required:
                coverage = len(matched) / len(required) * 100

                category_rows.append({
                    "Category": category,
                    "Required Skills": len(required),
                    "Matched Skills": len(matched),
                    "Coverage (%)": round(coverage, 1)
                })

        category_df = pd.DataFrame(category_rows)

        if not category_df.empty:

            category_fig = px.bar(
                category_df,
                x="Category",
                y="Coverage (%)",
                color="Category",
                text_auto=".1f",
                range_y=[0, 100],
                title="Skill Coverage by Category"
            )

            st.plotly_chart(
                category_fig,
                use_container_width=True
            )

            st.dataframe(
                category_df,
                use_container_width=True,
                hide_index=True
            )
        else:
            st.warning(
                "No skills from the current taxonomy were detected "
                "in the job description."
            )

    with tab2:

        st.subheader("Skills found in both documents")

        if matched_skills:
            st.write(", ".join(matched_skills))
        else:
            st.warning("No matching skills were detected.")

        st.subheader("Additional resume skills")

        if additional_skills:
            st.write(", ".join(additional_skills))
        else:
            st.write("No additional taxonomy skills detected.")

    with tab3:

        st.subheader("Potential skill gaps")

        if missing_skills:
            gap_df = pd.DataFrame({
                "Potential Gap": missing_skills,
                "Suggested Action": [
                    "If you have this skill, add genuine supporting evidence."
                    for _ in missing_skills
                ]
            })

            st.dataframe(
                gap_df,
                use_container_width=True,
                hide_index=True
            )

            st.write(
                "Review each gap carefully. A skill may be present in your "
                "resume but written using different terminology."
            )
        else:
            st.success("No potential skill gaps were detected.")

    with tab4:

        st.write(
            f"Extracted {len(resume_text):,} characters "
            f"from {page_count} page(s)."
        )

        st.text_area(
            "Extracted text",
            value=resume_text,
            height=350
        )

    # --------------------------------------------------
    # 11. DOWNLOAD REPORT
    # --------------------------------------------------

    report = f"""
RESUME–JOB DESCRIPTION MATCHING REPORT
=====================================

Resume file: {uploaded_file.name}
PDF pages: {page_count}

Text similarity: {text_similarity:.2f}%
Skill coverage: {skill_coverage:.2f}%
Overall matching indicator: {overall_score:.2f}%

MATCHED SKILLS
--------------
{', '.join(matched_skills) if matched_skills else 'None detected'}

POTENTIAL SKILL GAPS
--------------------
{', '.join(missing_skills) if missing_skills else 'None detected'}

ADDITIONAL RESUME SKILLS
------------------------
{', '.join(additional_skills) if additional_skills else 'None detected'}

DISCLAIMER
----------
Scores are informational indicators, not hiring predictions.
Skill extraction is rule-based and may miss skills or detect ambiguous terms.
"""

    st.download_button(
        "⬇️ Download Matching Report",
        data=report.encode("utf-8"),
        file_name="resume_matching_report.txt",
        mime="text/plain"
    )

else:

    st.markdown("### How it works")

    st.write(
        "1. Upload a PDF resume.\n"
        "2. Paste the target job description.\n"
        "3. Click Analyze Resume Match.\n"
        "4. Review the matching scores and potential skill gaps."
    )

st.caption(
    "ResumeMatch AI | Portfolio project | Candidate-support tool"
)
