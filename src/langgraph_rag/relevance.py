"""Calibrated, fail-closed relevance gate for retrieved evidence.

Why this exists
---------------
The retrieval surface exposes an RRF *rank* score. Measured on 2026-10-19, that
number does not separate an on-topic query from an irrelevant one (negative
control 0.0154-0.0164 vs on-topic 0.0159-0.0328), so it is not a relevance
signal and this module never reads it.

The gate consumes the raw per-candidate signals the retrieval layer computes
anyway (``keyword_score``, ``vector_similarity``, ``matched_terms``,
``term_coverage``). ``GATE_SIGNAL`` and ``DEFAULT_RELEVANCE_FLOOR`` were fitted
on a calibration split of a labelled query set *before* the held-out split was
measured; the artifacts under
``/root/.hermes/plans/langgraph-continuous-20261019/relevance/`` hold the
labelled set, the fitting rule, and the before/after numbers.

Fail-closed contract
--------------------
* No evidence -> the caller keeps the pre-existing outcomes (``no_evidence``
  for zero candidates, ``invalid_citations`` for ungroundable ones); this gate
  is not consulted.
* Evidence whose signal is missing, non-numeric, NaN/inf, or negative ->
  ``low_relevance``. Never ``ok``. That is what makes an old or broken
  retrieval producer fail closed instead of silently bypassing the gate.
* Signal present but below the floor -> ``low_relevance``.
"""
from __future__ import annotations

import math
from typing import Any

from .errors import Outcome

#: The calibrated raw signal used by the gate (fitted on the calibration split).
#: Per candidate this is the ``term_coverage`` field of the retrieval ``signals``
#: block; the response metadata echoes its aggregate as ``max_term_coverage``.
GATE_SIGNAL = "term_coverage"

#: Floor fitted on the calibration split; see the calibration artifacts.
#: Fitted 2026-10-19: min positive max_term_coverage (4/6 = 0.6667), rounded
#: down. All calibration negatives measured 0.0. The calibration record also
#: shows the cost: 3/35 labelled positives fall below this floor.
DEFAULT_RELEVANCE_FLOOR = 0.666666

#: Second calibrated signal (follow-up t_a518db57): matched-IDF mass.
#: The gate accepts evidence that clears *either* floor. Fitted on the same
#: calibration split by the same rule (first separating signal, floor = the
#: minimum positive value rounded down); see calibration-v2.json.
SECONDARY_SIGNAL = "idf_mass"
DEFAULT_IDF_FLOOR = 0.420043  # min calibration-positive idf_mass, rounded down
GATE_RULE = f"{GATE_SIGNAL} OR {SECONDARY_SIGNAL}"

MISSING_SIGNAL_REASON = "relevance signal missing or unusable"
BELOW_FLOOR_REASON = "no candidate cleared the calibrated relevance floor"


def _usable_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def _candidate_signal(item: Any, key: str = GATE_SIGNAL) -> float | None:
    if not isinstance(item, dict):
        return None
    signals = item.get("signals")
    if not isinstance(signals, dict):
        return None
    return _usable_number(signals.get(key))


def _max_signal(items: list, key: str) -> tuple[float | None, int]:
    value: float | None = None
    usable = 0
    for item in items:
        current = _candidate_signal(item, key)
        if current is None:
            continue
        usable += 1
        value = current if value is None else max(value, current)
    return value, usable


def evaluate(evidence: Any, *, floor: float | None = None,
             idf_floor: float | None = None) -> dict:
    """Evaluate the gate over already-validated evidence.

    Accepts when the best candidate clears *either* calibrated floor:
    ``term_coverage >= floor`` or ``idf_mass >= idf_floor``. Pure and
    deterministic: same evidence + floors -> same verdict.

    Fail-closed: evidence with no usable signal at all is ``low_relevance``,
    never ``ok``. A producer that emits only the primary signal keeps the
    primary gate exactly as before.
    """
    threshold = _usable_number(DEFAULT_RELEVANCE_FLOOR if floor is None else floor)
    if threshold is None:
        threshold = DEFAULT_RELEVANCE_FLOOR
    second_threshold = _usable_number(DEFAULT_IDF_FLOOR if idf_floor is None else idf_floor)
    if second_threshold is None:
        second_threshold = DEFAULT_IDF_FLOOR
    items = evidence if isinstance(evidence, list) else []

    signal, _ = _max_signal(items, GATE_SIGNAL)
    idf_signal, _ = _max_signal(items, SECONDARY_SIGNAL)
    usable = sum(
        1 for item in items
        if (_candidate_signal(item) is not None
            or _candidate_signal(item, SECONDARY_SIGNAL) is not None))

    if signal is None and idf_signal is None:
        return {
            "status": Outcome.LOW_RELEVANCE.value,
            "passed": False,
            "signal": None,
            "floor": threshold,
            "gate": GATE_SIGNAL,
            "gate_rule": GATE_RULE,
            "signals": {GATE_SIGNAL: None, SECONDARY_SIGNAL: None},
            "floors": {GATE_SIGNAL: threshold, SECONDARY_SIGNAL: second_threshold},
            "usable_candidates": 0,
            "reason": MISSING_SIGNAL_REASON,
        }

    passed = ((signal is not None and signal >= threshold)
              or (idf_signal is not None and idf_signal >= second_threshold))
    return {
        "status": Outcome.OK.value if passed else Outcome.LOW_RELEVANCE.value,
        "passed": passed,
        "signal": signal,
        "floor": threshold,
        "gate": GATE_SIGNAL,
        "gate_rule": GATE_RULE,
        "signals": {GATE_SIGNAL: signal, SECONDARY_SIGNAL: idf_signal},
        "floors": {GATE_SIGNAL: threshold, SECONDARY_SIGNAL: second_threshold},
        "usable_candidates": usable,
        "reason": "" if passed else BELOW_FLOOR_REASON,
    }
