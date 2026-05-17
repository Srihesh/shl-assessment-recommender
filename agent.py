from __future__ import annotations
import re

# out-of-scope triggers
OUT_OF_SCOPE_SIGNALS = [
    "salary", "compensation", "lawsuit", "gdpr", "discrimination",
    "how much does", "price", "cost", "discount",
    "ignore previous", "disregard", "you are now", "pretend you are",
    "forget your instructions", "ignore instructions", "jailbreak",
    "act as", "roleplay", "you are a",
    "weather", "politics", "sports", "stock", "crypto",
    "recipe", "restaurant", "movie", "music", "illegal", "regulation",
]

def is_out_of_scope(last_user_msg: str) -> bool:
    msg = last_user_msg.lower()
    return any(s in msg for s in OUT_OF_SCOPE_SIGNALS)

# role signals
ROLE_SIGNALS = [
    "developer", "engineer", "manager", "analyst", "designer", "architect",
    "consultant", "director", "executive", "officer", "lead", "head",
    "specialist", "coordinator", "administrator", "assistant", "associate",
    "sales", "marketing", "finance", "accounting", "hr", "human resources",
    "operations", "supply chain", "logistics", "compliance",
    "data scientist", "data analyst", "product manager", "project manager",
    "customer", "support", "service", "graduate", "intern", "junior",
    "senior", "mid", "entry", "experienced", "hiring", "recruit",
    "java", "python", "software", "tech", "research",
    "nurse", "doctor", "clinical", "healthcare", "banking", "insurance",
    "retail", "hospitality", "manufacturing", "construction",
]

COMPARE_SIGNALS = ["difference between", "compare", " vs ", " versus ", "which is better", "how do they differ", "what's the difference", "distinguish", "contrast"]
REFINE_SIGNALS  = ["actually", "instead", "add ", "remove ", "also include", "without ", "shorter", "longer", "change", "update", "exclude", "include", "drop ", "only ", "just ", "can you add", "can you remove", "more ", "fewer "]
SATISFIED_SIGNALS = ["thank", "thanks", "perfect", "great", "that's all", "that is all", "done", "good", "excellent", "helpful", "got it", "understood", "that's what i needed"]

def _all_user(messages: list) -> str:
    return " ".join(m.content for m in messages if m.role == "user").lower()

def _last_user(messages: list) -> str:
    return next((m.content for m in reversed(messages) if m.role == "user"), "").lower()

def _has_recs(messages: list) -> bool:
    return any(
        ('"recommendations"' in m.content and '"url"' in m.content)
        or "here are" in m.content.lower()
        or "i recommend" in m.content.lower()
        for m in messages if m.role == "assistant"
    )

def should_force_recommend(messages: list) -> bool:
    return len(messages) >= 6 and not _has_recs(messages)

def detect_mode(messages: list) -> str:
    last = _last_user(messages)

    if any(s in last for s in SATISFIED_SIGNALS) and _has_recs(messages):
        return "end"
    if any(s in last for s in COMPARE_SIGNALS):
        return "compare"
    if _has_recs(messages) and any(s in last for s in REFINE_SIGNALS):
        return "refine"
    if should_force_recommend(messages):
        return "recommend"

    has_role = any(re.search(r"\b" + re.escape(s.strip()) + r"\b", _all_user(messages)) for s in ROLE_SIGNALS)
    return "clarify" if not has_role else "recommend"

def build_retrieval_query(messages: list) -> str:
    user_msgs = [m.content for m in messages if m.role == "user"]
    return (" ".join(user_msgs) + " " + user_msgs[-1]) if user_msgs else ""

def format_history(messages: list) -> str:
    return "\n".join(("Hiring Manager" if m.role == "user" else "Advisor") + ": " + m.content for m in messages)
