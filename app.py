import os
import json
import mimetypes
import pandas as pd
import streamlit as st
import google.generativeai as genai

# ----------------------------
# Gemini Setup
# ----------------------------
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

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
def to_gemini_part(uploaded_file, debug=True):
    if debug:
        st.divider()
        st.subheader("📄 Attaching PDF CV to Gemini")

    if not uploaded_file.name.lower().endswith(".pdf"):
        st.error("❌ Only PDF CVs are supported at the moment.")
        raise ValueError("Non-PDF CV uploaded")

    uploaded_file.seek(0)

    data = uploaded_file.read()
    if not data:
        raise ValueError("PDF file is empty")

    if debug:
        st.write("Filename:", uploaded_file.name)
        st.write("File size (bytes):", len(data))

    return {
        "inline_data": {
            "mime_type": "application/pdf",
            "data": data,
        }
    }

#because ai output is consistent so we are making sure
def extract_json(text, debug=True): 
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
def generate_full_assessment(candidate_files, job, model_name, candidate_seniority):
    
    prompt = f"""
You MUST output a COMPLETE and VALID JSON object.
If you cannot finish, DO NOT start the response.
Do not omit closing braces.

Return ONLY valid JSON.
No markdown.
No text outside JSON.

あなたは、人材紹介エージェントとして
書類選考の実務経験が豊富なキャリアアドバイザーです。

以下の履歴書（CV）を読み、
「この候補者と求人要件の整合性を、書類情報の範囲で評価してください」
を、第三者にも説明できる形で評価してください。

【評価の前提（必ず厳守）】
- 本評価は「書類選考段階」の判断です
- CVに明示的に記載されている内容のみを根拠にしてください
- 推測・補完・好意的解釈は禁止です
- 求人票に記載された要件・文言を最重要視してください
- ENTRY求人では、経験不足を否定的に評価してはいけません
- 評価は「採用可否の最終判断」ではありません

【求人レベル】
{job["seniority"]}

【候補者レベル】
{candidate_seniority}

【評価対象の求人情報】
{job["job_context"][:1500]}

【評価の観点】
- 必須要件と経歴の適合性（最重要）
- 歓迎要件と経歴の適合性（加点要素）
- 職務内容全体との整合性
- 上記を踏まえた書類情報上の適合度


IMPORTANT:
- The JSON must start with {{ and end with }}
- Do not truncate output
- Do not stop mid-sentence

【出力JSON形式（厳守）】
{{
  "SUMMARY": "",
  "MUST_HAVE_REASONING": "",
  "PREFERRED_REASONING": "",
  "ROLE_ALIGNMENT_REASONING": "",
  "score": 0
}}
"""
    st.divider()
    st.subheader("🤖 Gemini Evaluation Call")

    st.write("Model name:", model_name)
    st.write("Candidate seniority:", candidate_seniority)
    st.write("Job title:", job["title"])
    st.write("Job seniority:", job["seniority"])
    st.write("Prompt length:", len(prompt))
    st.write("Number of attached CV files:", len(candidate_files))
    model = genai.GenerativeModel(model_name)

    contents = [prompt, *candidate_files]

    try:
        response = model.generate_content(
            contents,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": 900,
            }
        )
    except Exception as e:
        st.error("❌ Gemini API call failed")
        st.write("Model:", model_name)
        st.write("Job:", job["title"])
        st.code(repr(e))
        raise

    if not response.candidates:
        raise ValueError("No candidates returned by Gemini")
    
    candidate = response.candidates[0]
    
    if candidate.finish_reason != 0:
        raise ValueError(
            f"Gemini did not generate content. "
            f"finish_reason={candidate.finish_reason}"
        )
    
    if not candidate.content or not candidate.content.parts:
        raise ValueError("Gemini returned no content parts")
    
    text_parts = [
        p.text for p in candidate.content.parts if hasattr(p, "text")
    ]
    
    if not text_parts:
        raise ValueError("No text parts found in Gemini response")
    
    full_text = "\n".join(text_parts)
    
    return extract_json(full_text)

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

MODEL_NAME = "gemini-2.5-pro"






if uploaded_cvs and jobs_file and st.button("Evaluate CVs"):
    jobs_df = pd.read_excel(jobs_file)
    jobs = get_available_jobs(jobs_df)

    candidate_files = []
    for f in uploaded_cvs:
        candidate_files.append(to_gemini_part(f))

    candidate_seniority = "ENTRY"  # intentionally fixed (your original logic)

    st.subheader("📊 Results")

    results = []
    
    for job in jobs:
        try:
            result = generate_full_assessment(
                candidate_files,
                job,
                MODEL_NAME,
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
    results = sorted(
        results,
        key=lambda x: x["result"]["score"],
        reverse=True
    )
    
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
