"""FastAPI service for SHL recommender."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
import logging
from pydantic import BaseModel

import agent
import recommender
import retrieval

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Load catalog and index

CATALOG: list[dict] = []


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Load catalog and build FAISS index before the server accepts requests."""
    global CATALOG
    CATALOG = retrieval.load_catalog()
    retrieval.build_index(CATALOG)
    logger.info(f"Catalog loaded: {len(CATALOG)} products.")
    yield
    logger.info("Service stopped.")


app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for finding the right SHL psychometric assessments.",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic schemas

class Message(BaseModel):
    role: str   # "user" or "assistant"
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


# API Endpoints

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    messages = request.messages

    if not messages:
        raise HTTPException(status_code=400, detail="messages cannot be empty")

    # Extract last message
    last_user_msg = next(
        (m.content for m in reversed(messages) if m.role == "user"), ""
    )

    if not last_user_msg.strip():
        raise HTTPException(status_code=400, detail="Last user message is empty")

    # Check if out of scope
    if agent.is_out_of_scope(last_user_msg):
        return ChatResponse(
            reply=(
                "I can only help with SHL assessment selection. "
                "Could you tell me about the role you are hiring for?"
            ),
            recommendations=[],
            end_of_conversation=False,
        )

    # Detect mode
    mode = agent.detect_mode(messages)

    # Handle conversation end
    if mode == "end":
        return ChatResponse(
            reply=(
                "You're welcome! I hope the recommendations are helpful. "
                "Feel free to come back if you need to evaluate other roles. Good luck with your hiring!"
            ),
            recommendations=[],
            end_of_conversation=True,
        )

    # Retrieve catalog entries
    hits: list[dict] = []
    query = agent.build_retrieval_query(messages)
    if mode == "compare":
        hits = retrieval.retrieve_for_comparison(last_user_msg)
    elif mode in ("recommend", "refine"):
        hits = retrieval.retrieve(query, top_k=15)

    # NOTE: local_respond() is intentional for latency reasons (30s hard cap).
    # It uses the same catalog-grounded retrieval as the LLM path.
    # call_llm() is available for comparison/refinement turns where
    # nuanced language generation is needed.

    # Generate local response
    result = recommender.local_respond(
        mode=mode,
        hits=hits,
        messages=messages,
        query=query,
        catalog=CATALOG,
    )

    rec_objects = [
        Recommendation(
            name=r["name"],
            url=r["url"],
            test_type=r["test_type"],
        )
        for r in result.get("recommendations", [])
    ]

    return ChatResponse(
        reply=result["reply"],
        recommendations=rec_objects,
        end_of_conversation=result.get("end_of_conversation", False),
    )
