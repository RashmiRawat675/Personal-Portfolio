import os
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from groq import Groq
from pydantic import BaseModel
from pypdf import PdfReader
from docx import Document
import io

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("portfolio-assistant")

my_api_key = os.getenv("GROQ_API_KEY")
if not my_api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set")

client = Groq(api_key=my_api_key)

# Verify this exact model id is live on your Groq account before deploying —
# check https://console.groq.com/docs/models
MODEL = "openai/gpt-oss-20b"

# Resume file now lives next to this script instead of a hardcoded
# Windows path, so it works on any machine / any OS / any deploy target.
BASE_DIR = Path(__file__).resolve().parent
RESUME_PATH = BASE_DIR / "resume.pdf"

app = FastAPI(title="Portfolio AI Assistant")

@app.get("/test-groq")
async def test_groq():
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "user", "content": "Reply with only: OK"}
            ],
        )
        return {"status": "success", "response": response.choices[0].message.content}
    except Exception as e:
        return {
            "status": "error",
            "type": type(e).__name__,
            "error": str(e)
        }

# ---------------------------------------------------------------------------
# CORS — required so a browser-hosted frontend on a different origin/port
# is allowed to call this API. Lock allow_origins down to your real domain
# once you deploy (e.g. ["https://yourportfolio.com"]).
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class Experience(BaseModel):
    company_name: str | None = None
    role: str | None = None
    duration: str | None = None
    description: str | None = None
    skills_used: list[str] = []


class Resume(BaseModel):
    name: str | None = None
    email: str | None = None
    phone: str | None = None
    total_experience_years: float | None = None
    skills: list[str] = []
    experiences: list[Experience] = []
    projects: list[str] = []
    certifications: list[str] = []


resume_schema = Resume.model_json_schema()


class ChatMessage(BaseModel):
    role: str  # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[ChatMessage] = []  # optional — frontend can send prior turns
    job_description: str | None = None  # optional — set when HR attaches a JD


# ---------------------------------------------------------------------------
# Resume parsing / caching
# ---------------------------------------------------------------------------
def read_pdf(file_path: Path) -> str:
    if not file_path.exists():
        raise FileNotFoundError(f"Resume file not found at {file_path}")
    reader = PdfReader(str(file_path))
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def extract_text_from_upload(filename: str, raw: bytes) -> str:
    """Extract plain text from an uploaded JD file (pdf, docx, or txt)."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext == "pdf":
        reader = PdfReader(io.BytesIO(raw))
        return "".join(page.extract_text() or "" for page in reader.pages)
    if ext == "docx":
        doc = Document(io.BytesIO(raw))
        return "\n".join(p.text for p in doc.paragraphs)
    if ext == "txt":
        return raw.decode("utf-8", errors="ignore")
    raise ValueError(f"Unsupported file type: .{ext}")


def parse_resume(resume_text: str) -> Resume:
    system_prompt = f"""
    You are an expert resume parser.

    Extract information from the resume based on its meaning,
    not only exact section headings.

    Different resumes may use different headings, e.g.:
    - Experience / Professional Experience / Work History / Employment / Internship

    These may all contain relevant experience. Skills may appear in a
    dedicated skills section, or scattered across work experience,
    internships, or projects.

    Return only valid JSON matching this schema:
    {resume_schema}

    Rules:
    1. Do not invent information.
    2. If a value is not available, return null.
    3. If a list has no information, return an empty list.
    4. Include internships inside experiences.
    5. Extract skills mentioned across the entire resume.
    """
    user_prompt = f"Parse the following resume:\n{resume_text}"

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    data = json.loads(response.choices[0].message.content)
    return Resume(**data)


# Parsed once at startup and cached in memory — avoids re-reading the PDF
# and re-running an LLM parse on every single chat message.
resume_cache: Resume | None = None

# ---------------------------------------------------------------------------
# Personal / About Me profile
# ---------------------------------------------------------------------------
PERSONAL_PROFILE = {
    "personality": [
        "Curious about technology and likes understanding how things actually work.",
        "Prefers learning by doing rather than only studying theory.",
        "Persistent with problems and keeps trying different approaches until understanding what is wrong.",
        "Likes practical, honest feedback because it helps improve.",
        "Still developing communication skills, but actively tries to improve them.",
    ],

    "soft_skills": [
        "Problem-solving",
        "Willingness to learn",
        "Adaptability",
        "Persistence",
        "Taking feedback",
        "Self-learning",
        "Working independently",
        "Communication and explaining technical work in a simple way",
    ],

    "strengths": [
        "Can learn new technologies through hands-on practice.",
        "Doesn't easily give up when stuck on a technical problem.",
        "Likes breaking a problem into smaller parts and debugging step by step.",
        "Once a concept is understood, prefers implementing it in a real project.",
        "Comfortable exploring unfamiliar technologies when a project requires them.",
    ],

    "how_i_work": [
        "Prefers understanding the actual problem before jumping to a solution.",
        "When something doesn't work, debugs step by step rather than randomly changing everything.",
        "Likes building practical projects because they help understand concepts better.",
        "Prefers clear and direct feedback to correct mistakes quickly.",
    ],

    "how_i_learn": [
        "Learns best through hands-on projects, documentation, tutorials, and coding practice.",
        "Often starts with an example, implements it, encounters problems, then understands the concept more deeply while fixing them.",
        "Currently expanding beyond normal web development into AI engineering and LLM-based applications.",
    ],

    "career_goals": [
        "Primary goal is to start a career in a technical role working on real software/products.",
        "Interested in Full Stack, React, application development, and increasingly AI/LLM engineering.",
        "Wants to improve DSA, backend, system understanding, and AI engineering skills.",
        "Long term wants to become a strong software/AI engineer rather than limiting to one technology.",
    ],

    "interests": [
        "Building software projects",
        "Exploring new technologies",
        "Coding and problem solving",
        "AI/LLM applications",
        "Music",
        "Travelling and exploring places",
    ],
}

@app.on_event("startup")
def load_resume() -> None:
    global resume_cache
    try:
        resume_text = read_pdf(RESUME_PATH)
        resume_cache = parse_resume(resume_text)
        logger.info("Resume parsed and cached successfully.")
    except Exception as e:
        # Don't crash the whole app if the resume is missing/broken —
        # log it and let /chat report a clean error instead.
        logger.error(f"Failed to load resume at startup: {e}")
        resume_cache = None


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
def ask_candidate(
    question: str,
    resume: Resume,
    history: list[ChatMessage],
    job_description: str | None = None,
) -> str:
    jd_block = ""
    if job_description:
        jd_block = f"""
    A hiring manager has attached this job description:
    ---
    {job_description}
    ---
    When asked about fit, compare the JD's requirements against the candidate's
    actual skills, experience, and projects above. Be honest and specific:
    name what matches, name what's genuinely missing or weak, and give a direct
    verdict (e.g. strong fit / partial fit / not a fit) with reasoning. Do not
    inflate the match to sound impressive — an honest "not a fit" is more useful
    to the hiring manager than a flattering one.
    """
    system_prompt = f"""
    You are an AI assistant representing a job candidate.

    Below is the candidate's resume information:
    {resume.model_dump_json(indent=2)}

    Below is the candidate's personal and professional profile.
    Use this information for questions about personality, soft skills,
    strengths, work style, learning style, interests, and career goals:

    {json.dumps(PERSONAL_PROFILE, indent=2)}

    {jd_block}
    """

    messages = [{"role": "system", "content": system_prompt}]
    for turn in history:
        messages.append({"role": turn.role, "content": turn.content})
    messages.append({"role": "user", "content": question})

    stream = client.chat.completions.create(
    model=MODEL,
    messages=messages,
    stream=True
      )

    for chunk in stream:
          if chunk.choices:
            content = chunk.choices[0].delta.content

          if content:
            yield content


@app.get("/")
def home():
    return {"message": "Portfolio AI assistant is running.", "resume_loaded": resume_cache is not None}

@app.get("/resume")
def download_resume():
    if not RESUME_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail="Resume file not found."
        )

    return FileResponse(
        path=RESUME_PATH,
        media_type="application/pdf",
        filename="Rashmi_Rawat_Resume.pdf"
    )
@app.post("/chat")
def chat(request: ChatRequest):
    if resume_cache is None:
        raise HTTPException(
            status_code=503,
            detail="Resume data isn't loaded. Check server logs / GROQ_API_KEY / resume.pdf path.",
        )

    try:
        return StreamingResponse(
            ask_candidate(
                request.question,
                resume_cache,
                request.history,
                request.job_description
            ),
            media_type="text/plain"
        )

    except Exception as e:
        logger.error(f"Chat request failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong generating a response."
        )

@app.post("/parse-jd")
async def parse_jd(file: UploadFile = File(...)):
    """HR uploads a job description file; returns extracted plain text
    that the frontend then sends back as `job_description` on /chat."""
    try:
        raw = await file.read()
        text = extract_text_from_upload(file.filename, raw)
        if not text.strip():
            raise ValueError("No readable text found in that file.")
        return {"text": text}
    except Exception as e:
        logger.error(f"JD parsing failed: {e}")
        raise HTTPException(status_code=400, detail=str(e))