"""Deterministic turn eligibility for the rag-auto gate.

Every hard skip has a reason code from PLAN.md §5. The gate is intentionally
conservative: an uncertain turn is skipped, because a wrong automatic
retrieval costs latency and tokens on every turn, while a missed one costs one
explicit tool call.

Order (first match wins):

    profile_not_allowlisted -> empty_or_command_turn -> secret_or_sensitive
    -> privileged_or_destructive -> not_evidence_dependent (or trigger)

JEV is not consulted here. Phase 1 is deterministic only; an advisory JEV lane
would be an *upgrade* of the uncertain band, never an override of a hard skip.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

# -- reason codes (PLAN.md §5) ---------------------------------------------
PROFILE_NOT_ALLOWLISTED = "profile_not_allowlisted"
EMPTY_OR_COMMAND_TURN = "empty_or_command_turn"
SECRET_OR_SENSITIVE = "secret_or_sensitive"
PRIVILEGED_OR_DESTRUCTIVE = "privileged_or_destructive"
GRAPH_UNHEALTHY = "graph_unhealthy"
BUDGET_EXHAUSTED = "budget_exhausted"
DUPLICATE_TURN = "duplicate_turn"
NOT_EVIDENCE_DEPENDENT = "not_evidence_dependent"
TRIGGER = "trigger"

HARD_SKIP_REASONS = (
    PROFILE_NOT_ALLOWLISTED,
    EMPTY_OR_COMMAND_TURN,
    SECRET_OR_SENSITIVE,
    PRIVILEGED_OR_DESTRUCTIVE,
    GRAPH_UNHEALTHY,
    BUDGET_EXHAUSTED,
    DUPLICATE_TURN,
    NOT_EVIDENCE_DEPENDENT,
)


@dataclass(frozen=True)
class Decision:
    trigger: bool
    reason_code: str


def _skip(reason: str) -> Decision:
    return Decision(trigger=False, reason_code=reason)


TRIGGER_DECISION = Decision(trigger=True, reason_code=TRIGGER)


# -- secret / sensitive -----------------------------------------------------
_SECRET_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----",
    r"\bAKIA[0-9A-Z]{16}\b",
    r"\bgh[pousr]_[A-Za-z0-9]{20,}\b",
    r"\bsk-[A-Za-z0-9]{20,}\b",
    r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b",
    r"\b(api[_-]?key|secret|password|passwd|token|client[_-]?secret)\b\s*[:=]\s*\S{6,}",
    r"\bbearer\s+[A-Za-z0-9._~+/-]{20,}",
    r"\bmy (password|secret|api key|token) is\b",
    r"\b(here is|here's) (my|the) (password|secret|token|key)\b",
))

# -- privileged / destructive ----------------------------------------------
_PRIVILEGED_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE) for pattern in (
    r"\brm\s+(-[a-z]*[rf][a-z]*\s+)+\S",
    r"\bnixos-rebuild\b",
    r"\bsystemctl\s+(restart|stop|start|disable|enable|mask)\b",
    r"\b(kubectl|docker)\s+(delete|rm|down|prune)\b",
    r"\bdrop\s+(table|database|schema)\b",
    r"\bdelete\s+(the\s+)?(production|prod|live)\b",
    r"\bdelete\s+(the\s+)?(database|bucket|repository|repo|index)\b",
    r"\bdeploy\s+(to\s+)?(production|prod)\b",
    r"\bproduction\s+(database|server|deploy|deployment|credentials)\b",
    r"\b(transfer|send)\s+(funds|money|payment)\b",
    r"\bpay\s+(the\s+)?invoice\b",
    r"\b(sign|terminate)\s+(the\s+)?contract\b",
    r"\bsudo\b",
))

# -- positive knowledge intent ---------------------------------------------
_KNOWLEDGE_MARKERS = re.compile(
    r"\b(knowledge\s*base|corpus|rag|anytype|vault|notes?|docs?|documentation|polic(?:y|ies)|"
    r"runbook|spec(?:ification)?|decision\s*log|meeting\s*notes|changelog|tickets?|kanban|"
    r"project\s*state|source\s*of\s*truth|records?)\b",
    re.IGNORECASE,
)
_RECORDED_STATE = re.compile(
    r"\b(what\s+do\s+(we|i)\s+know|did\s+(we|i)\s+ever|have\s+(we|i)\s+ever|"
    r"where\s+did\s+(we|i)\s+(record|note|write|document)|what\s+did\s+(we|i)\s+decide|"
    r"(earlier|previously|last\s+time|prior)\b.{0,40}\b(decid|discuss|agree|record|note|mention))",
    re.IGNORECASE | re.DOTALL,
)
_SEARCH_IMPERATIVE = re.compile(
    r"\b(search|look\s*up|find|check|look)\b.{0,60}\b(notes?|docs?|documentation|knowledge\s*base|"
    r"corpus|vault|records?|anytype)\b",
    re.IGNORECASE | re.DOTALL,
)
_INTERROGATIVE = re.compile(
    r"^\s*(what|which|when|who|whom|whose|where|why|how|did|does|do|is|are|was|were|can|could|"
    r"should|would|has|have)\b",
    re.IGNORECASE,
)
_GREETING = re.compile(
    r"^(hi|hello|hey|yo|thanks|thank\s+you|ty|ok|okay|sure|cool|great|nice|good\s+morning|"
    r"good\s+afternoon|good\s+evening)\b[\s!.,?]*$",
    re.IGNORECASE,
)
_ARITHMETIC = re.compile(r"^[\d\s+\-*/().,=%^]+$")
_FORMATTING = re.compile(r"^(re)?format\s+(this|the\s+following|it)\b", re.IGNORECASE)


def text_of(user_message: object) -> str:
    """Text of a turn, supporting the multimodal list form the host may send."""
    if isinstance(user_message, str):
        return user_message
    if isinstance(user_message, (list, tuple)):
        parts: list[str] = []
        for part in user_message:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def fingerprint(text: str) -> str:
    """Stable hex digest of the normalised turn text (never stored raw)."""
    return hashlib.sha256(normalize(text).encode("utf-8")).hexdigest()[:24]


def _is_secret(text: str) -> bool:
    return any(pattern.search(text) for pattern in _SECRET_PATTERNS)


def _is_privileged(text: str) -> bool:
    return any(pattern.search(text) for pattern in _PRIVILEGED_PATTERNS)


def _is_trivial(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) <= 30 and _GREETING.match(stripped):
        return True
    if _ARITHMETIC.match(stripped):
        return True
    if "```" in stripped and "?" not in stripped and not _KNOWLEDGE_MARKERS.search(stripped):
        return True
    if len(stripped.split()) < 3 and "?" not in stripped and not _KNOWLEDGE_MARKERS.search(stripped):
        return True
    return False


def evaluate_turn(user_message: object) -> Decision:
    """First-match deterministic gate over the turn text."""
    text = text_of(user_message)
    stripped = text.strip()
    if not stripped or stripped.startswith("/"):
        return _skip(EMPTY_OR_COMMAND_TURN)
    if _is_secret(stripped):
        return _skip(SECRET_OR_SENSITIVE)
    if _is_privileged(stripped):
        return _skip(PRIVILEGED_OR_DESTRUCTIVE)
    question = "?" in stripped or bool(_INTERROGATIVE.match(stripped))
    if _RECORDED_STATE.search(stripped):
        return TRIGGER_DECISION
    if _SEARCH_IMPERATIVE.search(stripped):
        return TRIGGER_DECISION
    if _KNOWLEDGE_MARKERS.search(stripped) and (question or not _is_trivial(stripped)):
        if _FORMATTING.match(stripped) and "?" not in stripped:
            return _skip(NOT_EVIDENCE_DEPENDENT)
        return TRIGGER_DECISION
    return _skip(NOT_EVIDENCE_DEPENDENT)
