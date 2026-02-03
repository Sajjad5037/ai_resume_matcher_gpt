import os
import json
import mimetypes
import pandas as pd
import streamlit as st
from openai import OpenAI
import pdfplumber
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_pdf_text(uploaded_file):
    uploaded_file.seek(0)
    text = ""

    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

    if not text.strip():
        raise ValueError("No extractable text found in PDF")

    return text
# ----------------------------
# Gemini Setup
# ----------------------------


# ----------------------------
# App Config
# ----------------------------
st.set_page_config(page_title="AI Resume Matcher", layout="centered")

st.title("AI Resume Matcher")

st.write(
    "📌 For best accuracy, upload CVs in DOCX or text-based PDF format. "
    "Scanned PDFs may reduce matching quality."
)

# ----------------------------
# Helpers
# ----------------------------

#because ai output is consistent so we are making sure
def extract_json(text, debug=False): 
    if debug:
        st.divider()
        st.subheader("🧩 Parsing AI JSON Response")

    if debug:
        st.write("Raw response length:", len(text))
        st.write("Raw response preview:")
        st.code(text[:2000])  # prevent UI overload

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        st.error("❌ No valid JSON boundaries found in AI response")
        if debug:
            st.write("First '{' index:", start)
            st.write("Last '}' index:", end)
        raise ValueError("No valid JSON found in AI response")

    json_str = text[start:end + 1]

    if debug:
        st.success("JSON boundaries detected")
        st.write("Extracted JSON preview:")
        st.code(json_str[:2000])

    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        st.error("❌ JSON parsing failed")
        if debug:
            st.write("JSON decode error:")
            st.code(repr(e))
            st.write("Full extracted JSON:")
            st.code(json_str)
        raise

#here
def detect_seniority(job_context: str) -> str:
    keywords_entry = ["未経験OK", "経験不問", "第二新卒"]
    keywords_senior = ["3年以上", "5年以上", "リード", "マネージャー"]

    for k in keywords_entry:
        if k in job_context:
            return "ENTRY"

    for k in keywords_senior:
        if k in job_context:
            return "SENIOR"

    return "MID"


def get_available_jobs(df: pd.DataFrame):
    df.columns = df.columns.astype(str).str.strip()
    jobs = []

    def safe(v):
        return "" if pd.isna(v) else str(v).strip()

    for _, row in df.iterrows():
        job_context_parts = [
            f"【職種名】{safe(row.get('title'))}",
            f"【ポジション】{safe(row.get('position'))}",
            f"【業界】{safe(row.get('job_industry'))}",
            f"【勤務地】{safe(row.get('location'))}",
            f"【職務内容】{safe(row.get('job_content'))}",
        
            # 🔑 Make these unmistakable
            f"【必須要件（満たさない場合、原則書類通過不可）】{safe(row.get('required_experience'))}",
            f"【歓迎要件（加点要素）】{safe(row.get('desired_experience'))}",
        ]


        job_context = "\n".join(
            p for p in job_context_parts if p.strip()
        )


        jobs.append({
            "title": safe(row.get("title")) or "Unknown Role",
            "job_context": job_context,
            "seniority": detect_seniority(job_context),
            "company_name": safe(row.get("company_name")),
        })

    return jobs


# ----------------------------
# AI Core
# ----------------------------
def generate_full_assessment(candidate_texts, job, candidate_seniority):
    prompt = f"""
You MUST output a COMPLETE and VALID JSON object.
Return ONLY valid JSON.
No markdown.
No text outside JSON.

あなたは、人材紹介エージェントとして
書類選考の実務経験が豊富なキャリアアドバイザーです。

以下は【候補者の履歴書内容】です。
これを読み、【求人情報】との整合性を
書類情報の範囲で評価してください。

【候補者の履歴書】
{candidate_texts}

【求人レベル】
{job["seniority"]}

【評価対象の求人情報】
{job["job_context"]}

【評価の前提】
- 書類選考段階のみ
- 明示的記載のみを根拠
- 推測禁止
- ENTRY求人で経験不足を否定しない

【出力JSON形式（厳守）】
{{
  "SUMMARY": "",
  "MUST_HAVE_REASONING": "",
  "PREFERRED_REASONING": "",
  "ROLE_ALIGNMENT_REASONING": "",
  "score": 0
}}
"""

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": "You are a professional Japanese recruiter."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    text = response.choices[0].message.content
    return extract_json(text)

# ----------------------------
# UI
# ----------------------------
uploaded_cvs = st.file_uploader(
    "Upload CV files (PDF only for now)",
    type=["pdf"],
    accept_multiple_files=True
)

jobs_file = st.file_uploader(
    "Upload jobs Excel file",
    type=["xlsx"]
)








if uploaded_cvs and jobs_file and st.button("Evaluate CVs"):
    # ----------------------------
    # Load jobs
    # ----------------------------
    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    # ----------------------------
    # Candidate setup
    # ----------------------------
    candidate_seniority = "ENTRY"  # intentionally fixed for now

    # Extract ALL CV text ONCE (one candidate)
    candidate_texts = ""
    for f in uploaded_cvs:
        candidate_texts += extract_pdf_text(f) + "\n"

    st.subheader("📊 Results")

    # ----------------------------
    # Evaluate candidate against each job
    # ----------------------------
    results = []

    for job in jobs:
        try:
            result = generate_full_assessment(
                candidate_texts,
                job,
                candidate_seniority
            )

            results.append({
                "job": job,
                "result": result
            })

        except Exception as e:
            st.error(f"❌ Evaluation failed for job: {job['title']}")
            st.code(repr(e))
            continue

    # ----------------------------
    # Sort by score (descending)
    # ----------------------------
    results = sorted(
        results,
        key=lambda x: x["result"]["score"],
        reverse=True
    )
    st.success(f"✅ Evaluated {len(results)} jobs successfully")

    # ----------------------------
    # Display results
    # ----------------------------
    for item in results:
        job = item["job"]
        result = item["result"]

        st.markdown(f"### {job['title']}")
        st.write(f"**Score:** {result['score']}%")

        st.write("**Summary**")
        st.write(result["SUMMARY"])

        st.write("**Must Have**")
        st.write(result["MUST_HAVE_REASONING"])

        st.write("**Preferred**")
        st.write(result["PREFERRED_REASONING"])

        st.write("**Alignment**")
        st.write(result["ROLE_ALIGNMENT_REASONING"])

        st.divider()
