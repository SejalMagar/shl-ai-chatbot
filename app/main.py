from fastapi import FastAPI
from pydantic import BaseModel
from app.chat_engine import process_chat

app = FastAPI()

# Message schema
class Message(BaseModel):

    role: str
    content: str

# Request schema
class ChatRequest(BaseModel):

    messages: list[Message]

# Health endpoint
@app.get("/health")
def health():

    return {
        "status": "ok"
    }

# Chat endpoint
@app.post("/chat")
def chat(request: ChatRequest):

    response = process_chat(
        [m.dict() for m in request.messages]
    )

    return response