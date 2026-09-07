from __future__ import annotations

import json
import re

from pydantic import ValidationError

from app.semantic_analysis.models import SemanticParseStatus, SemanticResult


_JSON_FENCE = re.compile(r"^\s*```(?:json)?\s*(\{.*\})\s*```\s*$", re.IGNORECASE | re.DOTALL)


def parse_semantic_result(raw_text: str, expected_candidate_id: int) -> tuple[SemanticParseStatus, SemanticResult | None]:
    text = raw_text.strip()
    if not text:
        return SemanticParseStatus.EMPTY_RESPONSE, None
    candidates = [text]
    match = _JSON_FENCE.match(text)
    if match:
        candidates.append(match.group(1))
    # Conservative one-object extraction only when braces clearly bound one object.
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    seen: set[str] = set()
    had_json = False
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            payload = json.loads(candidate)
            had_json = True
        except json.JSONDecodeError:
            continue
        try:
            result = SemanticResult.model_validate(payload)
        except ValidationError:
            continue
        if result.candidate_id != expected_candidate_id:
            return SemanticParseStatus.SCHEMA_ERROR, None
        return SemanticParseStatus.SUCCESS, result
    return (SemanticParseStatus.SCHEMA_ERROR if had_json else SemanticParseStatus.INVALID_JSON), None
