import io
import json
import os
import re
import base64
import tempfile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from gtts import gTTS
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from pydantic import BaseModel
from pypdf import PdfReader

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="RAG API — Ollama + ChromaDB + HuggingFace Embeddings")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Singletons — loaded once at startup
# ---------------------------------------------------------------------------

_embeddings = HuggingFaceEmbeddings(model_name="keepitreal/vietnamese-sbert")
_vector_db: Chroma | None = None

# ---------------------------------------------------------------------------
# Chain-of-Thought RAG Prompt
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE = """ROLE: Bạn là trợ lý AI nghiêm túc, chỉ trả lời dựa trên TÀI LIỆU ĐƯỢC CUNG CẤP.

QUY TẮC TUYỆT ĐỐI:
1. Chỉ dùng thông tin trong phần Ngữ cảnh bên dưới.
2. Nếu không tìm thấy câu trả lời, phải nói trong thẻ <answer>: "Tôi không biết thông tin này vì không có trong tài liệu."
3. Không bịa, không đoán mò, không mở rộng ngoài phạm vi văn bản.

Ngữ cảnh:
{context}

Câu hỏi: {question}

Hãy suy nghĩ từng bước bên trong thẻ <thinking>:
- Xác định đoạn nào trong ngữ cảnh liên quan trực tiếp đến câu hỏi.
- Tổng hợp thông tin từ nhiều đoạn nếu cần, chú ý mâu thuẫn.
- Phác thảo câu trả lời dựa trên bằng chứng cụ thể.

Sau đó viết câu trả lời cuối cùng trong thẻ <answer>.

<thinking>
</thinking>
<answer>
</answer>"""

_PROMPT = PromptTemplate(template=_PROMPT_TEMPLATE, input_variables=["context", "question"])


def _extract_answer(raw: str) -> str:
    """Return the <answer> block content, falling back gracefully."""
    match = re.search(r"<answer>(.*?)</answer>", raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r"<answer>(.*?)$", raw, re.DOTALL)
    if match:
        text = match.group(1).strip()
        if text:
            return text
    cleaned = re.sub(r"<thinking>.*?</thinking>", "", raw, flags=re.DOTALL)
    return cleaned.strip() or raw.strip()


def _clean_mermaid(raw: str) -> str:
    """Strip markdown code fences the LLM sometimes wraps around Mermaid output."""
    cleaned = re.sub(r"```(?:mermaid)?\s*\n?", "", raw)
    return cleaned.replace("```", "").strip()


_GENERAL_TOPIC_KEYWORDS = frozenset({
    "tổng quan", "overview", "toàn bộ", "tất cả", "all topics",
    "summary", "tóm tắt", "comprehensive", "main topics", "toàn diện",
    "chủ đề chính", "nội dung chính", "document overview", "general",
    "full document", "entire document", "whole document",
})


def _is_general_topic(topic: str) -> bool:
    t = topic.lower().strip()
    return any(kw in t for kw in _GENERAL_TOPIC_KEYWORDS)


# ---------------------------------------------------------------------------
# Adaptive temperature routing
# ---------------------------------------------------------------------------

_CLASSIFY_PROMPT = """You are a strict query classifier. Given a DOCUMENT CONTEXT and a USER QUERY, classify the query into exactly one category. Output ONLY the integer 1, 2, or 3 — no explanation, no punctuation, no markdown.

Category 1: The query asks for direct facts, definitions, or strict content lookup that can be fully answered from the document context alone.
Category 2: The query is partially related to the document but expands or extends to external real-world knowledge or applications beyond the document.
Category 3: The query is completely unrelated to the document context.

DOCUMENT CONTEXT:
{context}

USER QUERY:
{query}

OUTPUT (1, 2, or 3):"""

_TEMPERATURE_MAP: dict[int, float] = {1: 0.0, 2: 0.3, 3: 0.5}


def _classify_query(context: str, query: str) -> float:
    classifier = Ollama(model="llama3.2", temperature=0.0)
    raw = classifier.invoke(_CLASSIFY_PROMPT.format(context=context, query=query)).strip()
    match = re.search(r"[123]", raw)
    category = int(match.group()) if match else 1
    return _TEMPERATURE_MAP.get(category, 0.0)


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    query: str


class GenerateQuizRequest(BaseModel):
    topic: str
    num_questions: int = 5
    difficulty: str = "Normal"


class MindmapRequest(BaseModel):
    topic: str


class SourceDoc(BaseModel):
    text: str
    cosine_score: float


class ChatResponse(BaseModel):
    answer: str
    audio_base64: str
    sources: list[SourceDoc]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {
        "status": "ok",
        "document_loaded": _vector_db is not None,
        "model": "llama3.2",
    }


@app.post("/upload", summary="Upload and index a PDF document")
async def upload_pdf(file: UploadFile = File(...)):
    global _vector_db

    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    content = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        reader = PdfReader(tmp_path)
        raw_text = "".join(page.extract_text() or "" for page in reader.pages)
    finally:
        os.remove(tmp_path)

    if not raw_text.strip():
        raise HTTPException(status_code=422, detail="No extractable text found in PDF.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=600, chunk_overlap=75, separators=["\n\n", "\n", " ", ""]
    ).split_text(raw_text)

    # hnsw:space=cosine → distance = 1 − cosine_similarity, inverted in /chat
    _vector_db = Chroma.from_texts(
        texts=chunks,
        embedding=_embeddings,
        collection_metadata={"hnsw:space": "cosine"},
    )

    return {"message": "Document indexed successfully.", "chunk_count": len(chunks)}


@app.post("/chat", response_model=ChatResponse, summary="RAG chat with CoT, cosine scores and TTS")
async def chat(request: ChatRequest):
    if _vector_db is None:
        raise HTTPException(status_code=400, detail="No document uploaded. POST to /upload first.")

    # ── Retrieval ────────────────────────────────────────────────────────────
    docs_and_scores = _vector_db.similarity_search_with_score(request.query, k=6)

    context = "\n\n".join(doc.page_content for doc, _ in docs_and_scores)
    sources = [
        SourceDoc(
            text=doc.page_content,
            cosine_score=round(1.0 - float(score), 4),
        )
        for doc, score in docs_and_scores
    ]

    # ── Intent classification → adaptive temperature ───────────────────────
    temperature = _classify_query(context, request.query)
    llm = Ollama(model="llama3.2", temperature=temperature)
    raw: str = llm.invoke(_PROMPT.format(context=context, question=request.query))
    answer = _extract_answer(raw)

    # ── TTS → Base64 ─────────────────────────────────────────────────────────
    try:
        tts = gTTS(text=answer, lang="vi", slow=False)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio_base64 = "data:audio/mp3;base64," + base64.b64encode(buf.read()).decode()
    except Exception:
        audio_base64 = ""

    return ChatResponse(answer=answer, audio_base64=audio_base64, sources=sources)


@app.post("/generate-quiz", summary="Generate a multiple-choice quiz from indexed document")
async def generate_quiz(request: GenerateQuizRequest):
    if _vector_db is None:
        raise HTTPException(status_code=400, detail="No document uploaded. POST to /upload first.")

    docs = _vector_db.similarity_search(request.topic, k=8)
    context = "\n\n".join(doc.page_content for doc in docs)

    difficulty_instruction = (
        f'DIFFICULTY / FOCUS DIRECTIVE: "{request.difficulty}". '
        "Strictly tailor every question to this direction. "
        "For 'Easy': use simple vocabulary, test recall of key definitions. "
        "For 'Hard': require multi-step reasoning, compare/contrast, or apply concepts. "
        "For any other value: treat it as a subject-focus filter and only ask about that aspect."
    )

    prompt = f"""You are a JSON-only quiz generator. Read the CONTEXT and produce exactly {request.num_questions} multiple-choice questions about "{request.topic}".

{difficulty_instruction}

STRICT RULES:
1. Output ONLY valid JSON. No commentary, no markdown fences, no text outside the JSON.
2. "correctAnswer" must be a 0-based integer index into that question's "options" array.
3. Every "explanation" must cite a fact stated directly in the CONTEXT — never invent.
4. Produce exactly {request.num_questions} questions inside the "questions" array.

--- EXAMPLE ---

CONTEXT:
Quang hợp là quá trình thực vật sử dụng ánh sáng mặt trời, nước và khí CO2 để tổng hợp glucose và giải phóng oxy. Quá trình này diễn ra bên trong lục lạp, nơi chứa chất diệp lục có khả năng hấp thụ ánh sáng.

TOPIC: Quang hợp
DIFFICULTY: Easy
NUM_QUESTIONS: 1

OUTPUT:
{{
  "title": "Quiz: Quang hợp",
  "questions": [
    {{
      "question": "Quang hợp diễn ra ở bào quan nào trong tế bào thực vật?",
      "options": ["Ti thể", "Lục lạp", "Nhân tế bào", "Không bào"],
      "correctAnswer": 1,
      "explanation": "Theo tài liệu, quang hợp diễn ra bên trong lục lạp — nơi chứa chất diệp lục hấp thụ ánh sáng mặt trời."
    }}
  ]
}}

--- YOUR TURN ---

CONTEXT:
{context}

TOPIC: {request.topic}
DIFFICULTY: {request.difficulty}
NUM_QUESTIONS: {request.num_questions}

OUTPUT:"""

    llm = Ollama(model="llama3.2", temperature=0.1, format="json")
    raw: str = llm.invoke(prompt)

    try:
        quiz_data = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="LLM produced invalid JSON. Please try again.")

    questions = quiz_data.get("questions")
    if not isinstance(questions, list) or len(questions) == 0:
        raise HTTPException(status_code=422, detail="LLM response missing valid 'questions' array.")

    for i, q in enumerate(questions):
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            raise HTTPException(status_code=422, detail=f"Question {i} is missing a 'question' string.")
        if not isinstance(q.get("options"), list) or len(q["options"]) < 2:
            raise HTTPException(status_code=422, detail=f"Question {i} must have at least 2 options.")
        if not isinstance(q.get("correctAnswer"), int):
            raise HTTPException(status_code=422, detail=f"Question {i} 'correctAnswer' must be an integer.")
        if q["correctAnswer"] < 0 or q["correctAnswer"] >= len(q["options"]):
            raise HTTPException(status_code=422, detail=f"Question {i} 'correctAnswer' index is out of range.")

    return quiz_data


@app.post("/generate-mindmap", summary="Generate a Mermaid.js mindmap from indexed document")
async def generate_mindmap(request: MindmapRequest):
    if _vector_db is None:
        raise HTTPException(status_code=400, detail="No document uploaded. POST to /upload first.")

    docs = _vector_db.similarity_search(request.topic, k=6)
    context = "\n\n".join(doc.page_content for doc in docs)

    if _is_general_topic(request.topic):
        prompt = f"""You are a Mermaid.js diagram generator. Your output must be ONLY valid Mermaid graph syntax.

CRITICAL RULES:
1. Output ONLY the raw Mermaid code. Absolutely no markdown backticks, no explanations, no prose.
2. Start the output directly with: graph TD
3. Keep node labels short (3-6 words). Wrap all labels in double quotes.
4. Use maximum 18 nodes. Prefer a clear 2-level hierarchy: root → pillars → key concepts.
5. Every edge must use the --> arrow.

TASK — DOCUMENT OVERVIEW MINDMAP:
The user wants a comprehensive top-level mindmap of the ENTIRE document.
Read the CONTEXT carefully and:
  1. Identify the 3-5 foundational pillars, chapters, or major themes present in the context.
  2. Under each pillar, list 2-3 of its most important sub-concepts or facts.
  3. Use a single root node labelled with the document's apparent subject.

--- EXAMPLE INPUT ---
CONTEXT: Chapter 1 covers Newton's three laws of motion. Chapter 2 discusses energy: kinetic, potential, and conservation. Chapter 3 explores waves: frequency, amplitude, and the electromagnetic spectrum. Forces such as friction, gravity, and tension are covered throughout.

--- EXAMPLE OUTPUT ---
graph TD
    ROOT["Physics Fundamentals"] --> P1["Newton's Laws of Motion"]
    ROOT --> P2["Energy"]
    ROOT --> P3["Waves & EM Spectrum"]
    ROOT --> P4["Forces"]
    P1 --> C1["Law 1: Inertia"]
    P1 --> C2["Law 2: F = ma"]
    P1 --> C3["Law 3: Action-Reaction"]
    P2 --> C4["Kinetic Energy"]
    P2 --> C5["Potential Energy"]
    P2 --> C6["Conservation of Energy"]
    P3 --> C7["Frequency & Amplitude"]
    P3 --> C8["Electromagnetic Spectrum"]
    P4 --> C9["Gravity & Friction"]
    P4 --> C10["Tension"]

--- YOUR TURN ---
CONTEXT:
{context}

OUTPUT:"""
    else:
        prompt = f"""You are a Mermaid.js diagram generator. Your output must be ONLY valid Mermaid graph syntax.

CRITICAL RULES:
1. Output ONLY the raw Mermaid code. Absolutely no markdown backticks, no explanations, no prose.
2. Start the output directly with: graph TD
3. Keep node labels short (3-5 words). Wrap labels containing special chars in double quotes.
4. Use maximum 15 nodes. Prefer a clear tree/hierarchy over a flat list.
5. Every edge must use the --> arrow.

--- EXAMPLE INPUT ---
TOPIC: Photosynthesis
CONTEXT: Photosynthesis occurs in chloroplasts using sunlight, CO2, and water to produce glucose and oxygen. It has two main stages: the light-dependent reactions and the Calvin cycle. Chlorophyll absorbs light energy.

--- EXAMPLE OUTPUT ---
graph TD
    A["Photosynthesis"] --> B["Location: Chloroplasts"]
    A --> C["Inputs"]
    A --> D["Outputs"]
    A --> E["Two Stages"]
    C --> F["Sunlight"]
    C --> G["CO2"]
    C --> H["Water"]
    D --> I["Glucose"]
    D --> J["Oxygen"]
    E --> K["Light Reactions"]
    E --> L["Calvin Cycle"]
    B --> M["Contains Chlorophyll"]

--- YOUR TURN ---
TOPIC: {request.topic}
CONTEXT:
{context}

OUTPUT:"""

    llm = Ollama(model="llama3.2", temperature=0.0)
    raw: str = llm.invoke(prompt)

    mermaid_code = _clean_mermaid(raw)

    if not mermaid_code.strip().startswith("graph"):
        lines = mermaid_code.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("graph"):
                mermaid_code = "\n".join(lines[i:])
                break
        else:
            raise HTTPException(
                status_code=422,
                detail="LLM did not produce valid Mermaid syntax. Please try again."
            )

    return {"mermaid_code": mermaid_code}
