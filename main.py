import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import httpx

app = FastAPI(title="Dermatology Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM_PROMPT = """أنت مساعد طبي متخصص حصراً في الأمراض الجلدية. اسمك "د. جلدية".
تخصصك: أمراض البشرة، الشعر، الأظافر، الفطريات الجلدية، العناية بالبشرة.
القواعد:
- لو السؤال مش عن الجلد أو الشعر أو الأظافر، اعتذر وقل أنك متخصص في الجلدية فقط.
- دائماً انصح بمراجعة طبيب متخصص للتشخيص الدقيق.
- لا تعطي جرعات دوائية — هذا من صلاحية الطبيب.
- تكلم بالعربية دائماً بأسلوب ودود ومهني."""


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: List[Message]


@app.get("/")
def root():
    return {"status": "ok", "service": "dermatology-chatbot"}


@app.post("/chat")
async def chat(request: ChatRequest):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m.role, "content": m.content} for m in request.messages]

    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            GROQ_URL,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {GROQ_API_KEY}",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": messages,
                "temperature": 0.4,
                "max_tokens": 800,
            },
        )

    data = response.json()
    reply = data.get("choices", [{}])[0].get("message", {}).get("content", "عذراً، حدث خطأ.")
    return {"reply": reply}
