from __future__ import annotations
import re
from collections import defaultdict

# type boost keywords
TYPE_BOOST_SIGNALS: dict[str, list[str]] = {
    "A": ["cognitive", "ability", "aptitude", "reasoning", "numerical", "verbal", "logical", "problem solving", "iq", "intelligence", "critical thinking", "abstract", "inductive", "deductive", "spatial", "mechanical", "graduate", "verify"],
    "P": ["personality", "behaviour", "behavior", "culture", "values", "attitude", "motivation", "work style", "fit", "opq", "character", "trait", "interpersonal", "communication", "leadership style", "soft skill"],
    "B": ["situational", "judgment", "judgement", "scenario", "biodata", "sjt", "situation", "decision", "real world"],
    "K": ["knowledge", "technical", "skill", "programming", "coding", "java", "python", "sql", "javascript", "software", "developer", "engineer", "it", "tech", ".net", "aws", "cloud", "data", "excel", "accounting", "finance", "legal", "medical", "nursing"],
    "S": ["simulation", "realistic", "hands-on", "practical", "data entry", "typing", "inbox", "exercise"],
    "E": ["exercise", "assessment centre", "assessment center", "in-tray", "group", "presentation", "role play"],
    "C": ["competency", "competencies", "360", "feedback", "development"],
    "D": ["360", "development", "coaching", "feedback"],
}

# level keywords
LEVEL_BOOST: dict[str, list[str]] = {
    "Director":                           ["director", "vp", "vice president", "c-suite", "cto", "ceo"],
    "Executive":                          ["executive", "c-level", "chief"],
    "Manager":                            ["manager", "management", "head of", "team lead"],
    "Front Line Manager":                 ["front line", "frontline", "supervisor", "team manager"],
    "Mid-Professional":                   ["mid", "middle", "5 year", "4 year", "3 year", "senior", "experienced"],
    "Professional Individual Contributor":["professional", "individual contributor", "specialist"],
    "Graduate":                           ["graduate", "entry", "junior", "new grad", "fresher", "intern", "trainee"],
    "General Population":                 [],
}

def _type_boosts(query: str) -> dict[str, float]:
    q = query.lower()
    out: dict[str, float] = defaultdict(float)
    for code, signals in TYPE_BOOST_SIGNALS.items():
        for s in signals:
            if re.search(r"\b" + re.escape(s) + r"\b", q):
                out[code] += 0.15
    return dict(out)

def _level_matches(query: str) -> set[str]:
    q = query.lower()
    return {lvl for lvl, sigs in LEVEL_BOOST.items() if any(s and s in q for s in sigs)}

def _duration_pref(query: str) -> str | None:
    q = query.lower()
    if any(s in q for s in ["quick", "short", "brief", "fast", "5 min", "10 min", "15 min", "under 20"]):
        return "short"
    if any(s in q for s in ["comprehensive", "detailed", "thorough", "full", "complete", "long"]):
        return "long"
    return None

def _parse_mins(dur: str) -> int | None:
    m = re.search(r"(\d+)", dur or "")
    return int(m.group(1)) if m else None

def _score(item: dict, type_boosts: dict, levels: set, dur_pref: str | None, wants_remote: bool) -> float:
    s = item.get("_score", 0.0)
    s += type_boosts.get(item.get("test_type", "K"), 0.0)
    item_lvls = set(item.get("competencies", []))
    if levels and item_lvls:
        s += len(levels & item_lvls) / max(len(levels), 1) * 0.1
    if dur_pref:
        mins = _parse_mins(item.get("duration", ""))
        if mins is not None:
            if dur_pref == "short" and mins <= 15: s += 0.08
            elif dur_pref == "long" and mins >= 20: s += 0.08
    if wants_remote and item.get("remote_testing"):
        s += 0.05
    return s

def rerank(hits: list[dict], query: str, n: int = 10) -> list[dict]:
    boosts   = _type_boosts(query)
    levels   = _level_matches(query)
    dur_pref = _duration_pref(query)
    remote   = any(s in query.lower() for s in ["remote", "online", "virtual", "work from home"])
    scored   = sorted(hits, key=lambda item: _score(item, boosts, levels, dur_pref, remote), reverse=True)
    return scored[:n]

def diversify(items: list[dict], max_same: int = 4) -> list[dict]:
    counts: dict[str, int] = defaultdict(int)
    out = []
    for item in items:
        t = item.get("test_type", "K")
        if counts[t] < max_same:
            out.append(item)
            counts[t] += 1
    return out

def _extract_role(messages: list) -> str:
    text = " ".join(m.content for m in messages if m.role == "user")
    for pat in [
        r"hir(?:ing|e) (?:a |an |for )?(.+?)(?:\.|,|$)",
        r"([a-z]+ (?:developer|engineer|manager|analyst|specialist|designer|consultant|officer|director|lead|associate))",
    ]:
        m = re.search(pat, text.lower())
        if m:
            role = m.group(1).strip()
            if 2 < len(role) < 60:
                return role.title()
    return "the role"

def _type_summary(items: list[dict]) -> str:
    labels = {"A": "cognitive ability", "P": "personality & behaviour", "B": "situational judgment", "K": "knowledge & skills", "S": "simulations", "E": "assessment exercises", "C": "competencies", "D": "development"}
    seen = list(dict.fromkeys(labels.get(i.get("test_type", "K"), "general") for i in items))
    return seen[0] if len(seen) == 1 else ", ".join(seen[:-1]) + " and " + seen[-1]

def generate_reply(mode: str, selected: list[dict], messages: list, refinement_note: str = "") -> str:
    role = _extract_role(messages)
    n    = len(selected)

    if mode == "clarify":
        return "I'd be happy to help. Could you tell me the job title or role you're hiring for?"

    if mode == "compare":
        return _compare_reply(selected)

    intro = (
        f"I've updated the shortlist{' — ' + refinement_note if refinement_note else ''}. Here are {n} assessment{'s' if n != 1 else ''} for {role}:"
        if mode == "refine" else
        f"Based on your requirements for a {role}, here are {n} SHL assessment{'s' if n != 1 else ''} I recommend:"
    )
    body = "\n".join(
        f"{i}. **{item['name']}**{' (' + item['duration'] + ')' if item.get('duration') else ''} — {item.get('test_type_full', item.get('test_type', ''))}"
        for i, item in enumerate(selected, 1)
    )
    closing = f"\nThese cover {_type_summary(selected)}. Let me know if you'd like to refine, compare, or get more detail."
    return intro + "\n" + body + closing

def _compare_reply(items: list[dict]) -> str:
    if not items:
        return "Could you name the assessments to compare? e.g. 'Compare OPQ and Verify G+'"
    lines = ["Here's a comparison from the SHL catalog:\n"]
    for item in items[:4]:
        lines.append(
            f"**{item['name']}** (Type: {item.get('test_type_full', '')})\n"
            f"  • Duration: {item.get('duration') or 'Not specified'}\n"
            f"  • Remote: {'Yes' if item.get('remote_testing') else 'No'}\n"
            f"  • Job levels: {', '.join(item.get('competencies', [])[:3]) or 'General'}\n"
            f"  • {item.get('description', '')[:200]}"
        )
    lines.append("\nWould you like a full shortlist based on any of these?")
    return "\n\n".join(lines)

def local_respond(mode: str, hits: list[dict], messages: list, query: str, catalog: list[dict]) -> dict:
    valid_urls = {item["url"] for item in catalog}

    if mode == "clarify":
        return {"reply": generate_reply("clarify", [], messages), "recommendations": [], "end_of_conversation": False}

    if mode == "compare":
        return {"reply": _compare_reply(hits[:4]), "recommendations": [], "end_of_conversation": False}

    ranked = rerank(hits, query, n=10)
    diverse = diversify(ranked, max_same=4)
    safe = [item for item in diverse if item.get("url") in valid_urls]

    recs = [{"name": item["name"], "url": item["url"], "test_type": item.get("test_type", "K")} for item in safe]

    note = ""
    if mode == "refine":
        note = next((m.content for m in reversed(messages) if m.role == "user"), "")[:80]

    return {"reply": generate_reply(mode, safe, messages, note), "recommendations": recs, "end_of_conversation": False}
