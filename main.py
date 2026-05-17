from __future__ import annotations
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import agent, recommender, retrieval

load_dotenv()

CATALOG: list[dict] = []

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    global CATALOG
    CATALOG = retrieval.load_catalog()
    retrieval.build_index(CATALOG)
    print(f"[startup] {len(CATALOG)} products loaded.")
    yield

app = FastAPI(title="SHL Recommender", version="1.0.0", lifespan=lifespan)

# schemas
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: list[Message]

class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str

class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool

# endpoints
@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages

    if not messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    last_user_msg = next((m.content for m in reversed(messages) if m.role == "user"), "")

    if not last_user_msg.strip():
        raise HTTPException(status_code=400, detail="empty user message")

    # scope check
    if agent.is_out_of_scope(last_user_msg):
        return ChatResponse(
            reply="I can only help with SHL assessment selection. Could you tell me about the role you are hiring for?",
            recommendations=[],
            end_of_conversation=False,
        )

    mode = agent.detect_mode(messages)

    # satisfied
    if mode == "end":
        return ChatResponse(
            reply="You're welcome! Feel free to come back if you need to evaluate other roles.",
            recommendations=[],
            end_of_conversation=True,
        )

    # retrieve
    hits: list[dict] = []
    query = agent.build_retrieval_query(messages)
    if mode == "compare":
        # extract named assessments from last user message and look up directly
        named = agent.extract_assessment_names(last_user_msg)
        hits = [h for n in named for h in [retrieval.retrieve_by_name(n)] if h]
        if not hits:
            hits = retrieval.retrieve(query, top_k=6)
    elif mode in ("recommend", "refine"):
        hits = retrieval.retrieve(query, top_k=15)

    # recommend
    result = recommender.local_respond(mode=mode, hits=hits, messages=messages, query=query, catalog=CATALOG)

    recs = [Recommendation(name=r["name"], url=r["url"], test_type=r["test_type"]) for r in result.get("recommendations", [])]

    return ChatResponse(reply=result["reply"], recommendations=recs, end_of_conversation=result.get("end_of_conversation", False))
