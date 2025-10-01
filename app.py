import os
import io
import re
import streamlit as st
import PyPDF2
from docx import Document
from huggingface_hub import InferenceClient, login
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google_auth_oauthlib.flow import InstalledAppFlow
import google.auth

# =====================
# CONFIG & AUTH
# =====================
SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]

# Hugging Face login (read token from secrets/environment)
HF_TOKEN = os.getenv("HF_TOKEN", None)
if HF_TOKEN:
    login(token=HF_TOKEN)
else:
    st.error("❌ Missing Hugging Face token. Please set HF_TOKEN in Streamlit secrets or env vars.")

# =====================
# DRIVE UTILS
# =====================

ALLOWED_MIMES = [
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "text/plain",
    "text/csv",
    "application/vnd.google-apps.document",
    "application/vnd.google-apps.spreadsheet",
]

def sanitize_for_drive(q: str) -> str:
    q = q.replace("'", " ").replace('"', " ").replace("\\", " ")
    q = re.sub(r"[^0-9A-Za-z\s]", " ", q)
    q = re.sub(r"\s+", " ", q).strip()
    return q

def keywords_from_query(q: str, min_len: int = 2):
    s = sanitize_for_drive(q)
    return [t for t in s.split() if len(t) >= min_len]

def build_drive_q_for_keywords(keywords: list):
    if not keywords:
        return None
    parts = [f"(name contains '{k}' or fullText contains '{k}')" for k in keywords]
    keyword_filter = " or ".join(parts)
    mime_filter = " or ".join([f"mimeType='{m}'" for m in ALLOWED_MIMES])
    return f"({keyword_filter}) and ({mime_filter}) and trashed=false"

def read_file(file_id, mime_type, drive_service):
    fh = io.BytesIO()
    if mime_type.startswith("application/vnd.google-apps"):
        if mime_type == "application/vnd.google-apps.document":
            request = drive_service.files().export_media(fileId=file_id, mimeType="text/plain")
        elif mime_type == "application/vnd.google-apps.spreadsheet":
            request = drive_service.files().export_media(fileId=file_id, mimeType="text/csv")
        else:
            return f"[Skipped unsupported Google file type: {mime_type}]"
    else:
        request = drive_service.files().get_media(fileId=file_id)

    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    fh.seek(0)

    if mime_type == "application/pdf":
        reader = PyPDF2.PdfReader(fh)
        return " ".join([p.extract_text() for p in reader.pages if p.extract_text()])
    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = Document(fh)
        return " ".join([p.text for p in doc.paragraphs])
    else:
        return fh.read().decode("utf-8", errors="ignore")

def search_drive(query, drive_service, max_results=5):
    keywords = keywords_from_query(query)
    if not keywords:
        return []

    q = build_drive_q_for_keywords(keywords)
    try:
        res = drive_service.files().list(
            q=q,
            pageSize=max_results,
            fields="files(id, name, mimeType, size, modifiedTime)",
            orderBy="modifiedTime desc",
        ).execute()
        return res.get("files", []) or []
    except Exception as e:
        st.error(f"Drive search error: {e}")
        return []

# =====================
# CHAT AGENT
# =====================
class DriveChatAgent:
    def __init__(self, model="meta-llama/Meta-Llama-3-8B-Instruct"):
        self.client = InferenceClient(model, token=HF_TOKEN)

    def answer_question(self, user_query, drive_service):
        files = search_drive(user_query, drive_service)

        context = ""
        if files:
            for f in files:
                try:
                    text = read_file(f["id"], f["mimeType"], drive_service)
                    snippet = text[:5000]
                    context += f"\n--- {f['name']} ---\n{snippet}"
                except Exception as e:
                    context += f"\n--- {f['name']} ---\n[Error: {e}]"

        system = (
            "You are an assistant that answers user questions using provided context from Google Drive. "
            "If the context is empty, use your own knowledge."
        )

        user_text = f"Context:\n{context}\n\nQuestion: {user_query}"

        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user_text},
            ],
            max_tokens=500,
            temperature=0.2,
        )

        return response.choices[0].message["content"].strip()

# =====================
# STREAMLIT UI
# =====================
st.title("📂 Google Drive + Hugging Face Chatbot")

st.write("Ask a question and I will search your Google Drive for answers.")

query = st.text_input("Enter your question:")

if st.button("Search & Answer"):
    if not query.strip():
        st.warning("Please enter a question first.")
    else:
        try:
            # Google Auth flow
            flow = InstalledAppFlow.from_client_secrets_file("cred.json", SCOPES)
            creds = flow.run_local_server(port=0)
            drive_service = build("drive", "v3", credentials=creds)

            st.info("✅ Authenticated with Google Drive")

            agent = DriveChatAgent()
            answer = agent.answer_question(query, drive_service)

            st.subheader("Answer:")
            st.write(answer)

        except Exception as e:
            st.error(f"❌ Error: {e}")
