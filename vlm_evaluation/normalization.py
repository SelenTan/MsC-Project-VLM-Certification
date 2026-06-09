#!/usr/bin/env python3
"""Normalize dataset references and numeric values for checker calibration."""

from __future__ import annotations

import json
import re
from typing import Any


NUMBER_PATTERN = re.compile(r"(?<![A-Za-z0-9])(?:Rp\.?\s*)?[$€£¥]?-?\d(?:[\d,.\s]*\d)?%?")


def parse_json_like(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text or text[0] not in "[{":
        return value
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return value


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        normalized = " ".join(str(value).split())
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)
    return result


def normalize_reference_answer(value: Any) -> str:
    """Convert heterogeneous expected answers into a compact checker-facing reference string."""
    parsed = parse_json_like(value)
    if isinstance(parsed, str) and '"words"' in parsed:
        words = re.findall(r'"([^"]+)"', parsed)
        words = [word for word in words if word != "words"]
        if words:
            return "visible_words: " + "; ".join(unique_preserve_order(words))
    if isinstance(parsed, list):
        full_answers: list[str] = []
        ui_texts: list[str] = []
        scalar_values: list[str] = []
        for entry in parsed:
            if isinstance(entry, dict):
                if entry.get("full_answer") is not None:
                    full_answers.append(str(entry["full_answer"]))
                for element in entry.get("ui_elements") or []:
                    if isinstance(element, dict) and element.get("text") is not None:
                        ui_texts.append(str(element["text"]))
            else:
                scalar_values.append(str(entry))
        parts: list[str] = []
        if full_answers:
            parts.append("answers: " + "; ".join(unique_preserve_order(full_answers)))
        if ui_texts:
            parts.append("visible_ui_text: " + "; ".join(unique_preserve_order(ui_texts)))
        if scalar_values:
            parts.append("; ".join(unique_preserve_order(scalar_values)))
        return " | ".join(parts) if parts else str(value)
    if isinstance(parsed, dict):
        if isinstance(parsed.get("words"), list):
            return "visible_words: " + "; ".join(unique_preserve_order([str(word) for word in parsed["words"]]))
        if parsed.get("full_answer") is not None:
            return str(parsed["full_answer"])
        return json.dumps(parsed, ensure_ascii=False, sort_keys=True)
    return " ".join(str(value).split())


def normalize_reference_evidence(value: Any) -> str:
    """Keep evidence descriptions concise and whitespace-normalized for dataset consistency."""
    return " ".join(str(value or "").split())


def normalize_number_token(token: str) -> list[str]:
    cleaned = token.strip()
    cleaned = re.sub(r"(?i)rp\.?", "", cleaned)
    cleaned = re.sub(r"[$€£¥%A-Za-z]", "", cleaned)
    cleaned = re.sub(r"\s+", "", cleaned)
    cleaned = cleaned.strip(".,")
    if not cleaned or not any(character.isdigit() for character in cleaned):
        return []

    candidates = [cleaned]
    if "," in cleaned and "." in cleaned:
        last_comma = cleaned.rfind(",")
        last_dot = cleaned.rfind(".")
        if last_comma > last_dot:
            candidates.append(cleaned.replace(".", "").replace(",", "."))
        else:
            candidates.append(cleaned.replace(",", ""))
    elif "," in cleaned:
        parts = cleaned.split(",")
        if len(parts[-1]) == 3 and all(part.isdigit() for part in parts):
            candidates.append("".join(parts))
        else:
            candidates.append(cleaned.replace(",", "."))
        candidates.append(cleaned.replace(",", ""))
    elif "." in cleaned:
        parts = cleaned.split(".")
        if len(parts[-1]) == 3 and all(part.isdigit() for part in parts):
            candidates.append("".join(parts))
        else:
            candidates.append(cleaned)

    normalized: list[str] = []
    for candidate in candidates:
        if "." in candidate and "," not in candidate:
            candidate = candidate.rstrip("0").rstrip(".")
        if candidate == "-0":
            candidate = "0"
        normalized.append(candidate)
    return unique_preserve_order(normalized)


def extract_normalized_values(text: Any) -> list[str]:
    """Extract normalized numeric values from text while preserving likely equivalent formats."""
    source = normalize_reference_answer(text) if not isinstance(text, str) else text
    values: list[str] = []
    for match in NUMBER_PATTERN.finditer(source):
        values.extend(normalize_number_token(match.group(0)))
    return unique_preserve_order(values)


def build_checker_reference_fields(item: dict[str, Any]) -> dict[str, Any]:
    reference_answer = normalize_reference_answer(item.get("expected_answer_or_behavior"))
    reference_evidence = normalize_reference_evidence(item.get("expected_evidence"))
    return {
        "checker_reference_answer": reference_answer,
        "checker_reference_evidence": reference_evidence,
        "checker_reference_values": extract_normalized_values(reference_answer),
    }


def build_checker_normalization(record: dict[str, Any]) -> dict[str, Any]:
    reference_values = record.get("checker_reference_values") or extract_normalized_values(
        record.get("checker_reference_answer") or record.get("expected_answer_or_behavior")
    )
    response_values = extract_normalized_values(record.get("target_model_response") or "")
    overlap = sorted(set(reference_values).intersection(response_values))
    if overlap:
        hint = f"normalized numeric values match: {', '.join(overlap)}"
    elif reference_values and response_values:
        hint = "normalized numeric values differ"
    elif reference_values:
        hint = "reference contains numeric values but response has no extracted numeric value"
    else:
        hint = "no numeric normalization hint"
    return {
        "checker_reference_values": reference_values,
        "target_response_values": response_values,
        "normalization_hint": hint,
    }
