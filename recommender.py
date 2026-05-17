"""Local recommendation engine."""

from __future__ import annotations

import re
from collections import defaultdict

# Keyword to test_type boost signals

TYPE_BOOST_SIGNALS: dict[str, list[str]] = {
    "A": [  # Ability & Aptitude
        "cognitive", "ability", "aptitude", "reasoning", "numerical",
        "verbal", "logical", "problem solving", "iq", "intelligence",
        "critical thinking", "abstract", "inductive", "deductive",
        "spatial", "mechanical", "graduate", "verify",
    ],
    "P": [  # Personality & Behavior
        "personality", "behaviour", "behavior", "culture", "values",
        "attitude", "motivation", "work style", "fit", "opq",
        "character", "trait", "interpersonal", "communication",
        "leadership style", "soft skill",
    ],
    "B": [  # Biodata & Situational Judgment
        "situational", "judgment", "judgement", "scenario", "biodata",
        "sjt", "situation", "decision", "real world",
    ],
    "K": [  # Knowledge & Skills
        "knowledge", "technical", "skill", "programming", "coding",
        "java", "python", "sql", "javascript", "software", "developer",
        "engineer", "it", "tech", ".net", "aws", "cloud", "data",
        "excel", "accounting", "finance", "legal", "medical", "nursing",
    ],
    "S": [  # Simulations
        "simulation", "realistic", "hands-on", "practical", "data entry",
        "typing", "inbox", "exercise",
    ],
    "E": [  # Assessment Exercises
        "exercise", "assessment centre", "assessment center", "in-tray",
        "group", "presentation", "role play",
    ],
    "C": [  # Competencies
        "competency", "competencies", "360", "feedback", "development",
    ],
    "D": [  # Development & 360
        "360", "development", "coaching", "feedback",
    ],
}

# Seniority level boost signals
LEVEL_BOOST: dict[str, list[str]] = {
    "Director":                    ["director", "vp", "vice president", "c-suite", "cto", "ceo"],
    "Executive":                   ["executive", "c-level", "chief"],
    "Manager":                     ["manager", "management", "head of", "team lead"],
    "Front Line Manager":          ["front line", "frontline", "supervisor", "team manager"],
    "Mid-Professional":            ["mid", "middle", "5 year", "4 year", "3 year", "senior", "experienced"],
    "Professional Individual Contributor": ["professional", "individual contributor", "specialist"],
    "Graduate":                    ["graduate", "entry", "junior", "new grad", "fresher", "intern", "trainee"],
    "General Population":          [],
}

# Duration preference signals
DURATION_SIGNALS = {
    "short":  ["quick", "short", "brief", "fast", "5 min", "10 min", "15 min", "under 20"],
    "long":   ["comprehensive", "detailed", "thorough", "full", "complete", "long"],
}

# Remote preference
REMOTE_SIGNALS = ["remote", "online", "virtual", "work from home", "distributed"]


# Scoring logic

def _extract_type_boosts(query: str) -> dict[str, float]:
    """Return boost values per test_type based on keywords in query."""
    q = query.lower()
    boosts: dict[str, float] = defaultdict(float)
    for type_code, signals in TYPE_BOOST_SIGNALS.items():
        for s in signals:
            if re.search(r"\b" + re.escape(s) + r"\b", q):
                boosts[type_code] += 0.15
    return dict(boosts)


def _extract_level_boosts(query: str) -> set[str]:
    """Return which job levels are relevant based on query keywords."""
    q = query.lower()
    matched: set[str] = set()
    for level, signals in LEVEL_BOOST.items():
        for s in signals:
            if s and s in q:
                matched.add(level)
    return matched


def _duration_preference(query: str) -> str | None:
    """Return 'short', 'long', or None."""
    q = query.lower()
    if any(s in q for s in DURATION_SIGNALS["short"]):
        return "short"
    if any(s in q for s in DURATION_SIGNALS["long"]):
        return "long"
    return None


def _wants_remote(query: str) -> bool:
    q = query.lower()
    return any(s in q for s in REMOTE_SIGNALS)


def _parse_duration_minutes(duration_str: str) -> int | None:
    """Parse '15 minutes' → 15, return None if unparseable."""
    m = re.search(r"(\d+)", duration_str or "")
    return int(m.group(1)) if m else None


def score_hit(
    item: dict,
    type_boosts: dict[str, float],
    matched_levels: set[str],
    duration_pref: str | None,
    wants_remote: bool,
) -> float:
    score: float = item.get("_score", 0.0)  # base = FAISS cosine similarity

    # Test-type boost
    score += type_boosts.get(item.get("test_type", "K"), 0.0)

    # Job-level match boost
    item_levels = set(item.get("competencies", []))
    if matched_levels and item_levels:
        overlap = len(matched_levels & item_levels) / max(len(matched_levels), 1)
        score += overlap * 0.1

    # Duration preference
    if duration_pref:
        mins = _parse_duration_minutes(item.get("duration", ""))
        if mins is not None:
            if duration_pref == "short" and mins <= 15:
                score += 0.08
            elif duration_pref == "long" and mins >= 20:
                score += 0.08

    # Remote preference
    if wants_remote and item.get("remote_testing"):
        score += 0.05

    return score


def rerank(hits: list[dict], query: str, n: int = 10) -> list[dict]:
    """
    Re-score FAISS hits using hybrid signals and return top-n.
    """
    type_boosts = _extract_type_boosts(query)
    matched_levels = _extract_level_boosts(query)
    duration_pref = _duration_preference(query)
    wants_remote = _wants_remote(query)

    scored = [
        (score_hit(item, type_boosts, matched_levels, duration_pref, wants_remote), item)
        for item in hits
    ]
    scored.sort(key=lambda x: x[0], reverse=True)

    return [item for _, item in scored[:n]]


# Diversity filter

def diversify(items: list[dict], max_same_type: int = 4) -> list[dict]:
    """
    Limit consecutive items of the same test_type to keep shortlist varied.
    """
    type_counts: dict[str, int] = defaultdict(int)
    result = []
    for item in items:
        t = item.get("test_type", "K")
        if type_counts[t] < max_same_type:
            result.append(item)
            type_counts[t] += 1
    return result


# Reply template generator

def _extract_role(messages: list) -> str:
    """Best-effort role extraction from conversation history."""
    all_text = " ".join(m.content for m in messages if m.role == "user")
    # Common role patterns
    patterns = [
        r"hir(?:ing|e) (?:a |an |for )?(.+?)(?:\.|,|$)",
        r"(?:role|position|job) (?:is |of |for )?(.+?)(?:\.|,|$)",
        r"([a-z\s]+ (?:role|position|job))",
        r"(?:a |an |for |our )([a-z]+ (?:developer|engineer|manager|analyst|specialist|designer|consultant|officer|director|lead|associate))",
        r"([a-z]+ (?:developer|engineer|manager|analyst|specialist|designer|consultant|officer|director|lead|associate))",
    ]
    for pat in patterns:
        m = re.search(pat, all_text.lower())
        if m:
            role = m.group(1).strip()
            if 2 < len(role) < 60 and role not in ("the", "a", "an"):
                return role.title()
                
    if "leadership" in all_text.lower() or "managing" in all_text.lower() or "manager" in all_text.lower() or "team" in all_text.lower():
        return "Leadership/Management Role"
        
    return "the role"


def generate_reply(
    mode: str,
    selected: list[dict],
    messages: list,
    refinement_note: str = "",
) -> str:
    role = _extract_role(messages)
    n = len(selected)

    if mode == "clarify":
        return (
            "I'd be happy to help you find the right SHL assessment. "
            "Could you tell me the job title or role you're hiring for?"
        )

    if mode == "compare":
        return _compare_reply(selected)

    # recommend / refine
    type_summary = _type_summary(selected)

    if mode == "refine":
        intro = (
            f"I've updated the shortlist based on your preferences"
            f"{' — ' + refinement_note if refinement_note else ''}. "
            f"Here are {n} assessment{'s' if n != 1 else ''} for {role}:"
        )
    else:
        intro = (
            f"Based on your requirements for a {role}, "
            f"here are {n} SHL assessment{'s' if n != 1 else ''} I recommend:"
        )

    body_lines = []
    for i, item in enumerate(selected, 1):
        dur = f" ({item['duration']})" if item.get("duration") else ""
        body_lines.append(f"{i}. **{item['name']}**{dur} — {item.get('test_type_full', item.get('test_type', ''))}")

    closing = (
        f"\nThese cover {type_summary}. "
        "Let me know if you'd like to refine the list, compare any two assessments, or need more detail on a specific one."
    )

    return intro + "\n" + "\n".join(body_lines) + closing


def _type_summary(items: list[dict]) -> str:
    type_full = {
        "A": "cognitive ability", "P": "personality & behaviour",
        "B": "situational judgment", "K": "knowledge & skills",
        "S": "simulations", "E": "assessment exercises",
        "C": "competencies", "D": "development",
    }
    seen_types = list(dict.fromkeys(
        type_full.get(item.get("test_type", "K"), "general") for item in items
    ))
    if len(seen_types) == 1:
        return seen_types[0]
    return ", ".join(seen_types[:-1]) + " and " + seen_types[-1]


def _compare_reply(items: list[dict]) -> str:
    if not items:
        return (
            "Could you name the specific SHL assessments you'd like to compare? "
            "For example: 'Compare OPQ and Verify G+'"
        )
    lines = ["Here's a comparison based on the SHL catalog:\n"]
    for item in items[:4]:
        dur = item.get("duration") or "Not specified"
        remote = "Yes" if item.get("remote_testing") else "No"
        levels = ", ".join(item.get("competencies", [])[:3]) or "General"
        lines.append(
            f"**{item['name']}** (Type: {item.get('test_type_full', '')})\n"
            f"  • Duration: {dur}\n"
            f"  • Remote: {remote}\n"
            f"  • Job levels: {levels}\n"
            f"  • {item.get('description', '')[:200]}"
        )
    lines.append(
        "\nWould you like me to recommend a full shortlist based on any of these?"
    )
    return "\n\n".join(lines)


# Main entry point

def local_respond(
    mode: str,
    hits: list[dict],
    messages: list,
    query: str,
    catalog: list[dict],
) -> dict:
    """
    Drop-in replacement for agent.call_llm().
    Returns: {reply, recommendations, end_of_conversation}
    """
    valid_urls = {item["url"] for item in catalog}

    if mode == "clarify":
        return {
            "reply": generate_reply("clarify", [], messages),
            "recommendations": [],
            "end_of_conversation": False,
        }

    if mode == "compare":
        # Use top hits for comparison
        compare_items = hits[:4]
        return {
            "reply": _compare_reply(compare_items),
            "recommendations": [],
            "end_of_conversation": False,
        }

    # recommend / refine
    ranked = rerank(hits, query, n=10)
    diverse = diversify(ranked, max_same_type=4)

    # Hard URL safety — only catalog-verified items
    safe = [item for item in diverse if item.get("url") in valid_urls]

    # Build recommendations list
    recommendations = [
        {
            "name": item["name"],
            "url": item["url"],
            "test_type": item.get("test_type", "K"),
        }
        for item in safe
    ]

    # Extract refinement note if mode is refine
    refinement_note = ""
    if mode == "refine":
        last_user = next(
            (m.content for m in reversed(messages) if m.role == "user"), ""
        )
        refinement_note = last_user[:80]

    reply = generate_reply(mode, safe, messages, refinement_note)

    return {
        "reply": reply,
        "recommendations": recommendations,
        "end_of_conversation": False,
    }
