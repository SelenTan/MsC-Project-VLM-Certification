#!/usr/bin/env python3
"""Call a local text-only AI checker for VLM certification labels."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Optional
from urllib import error, request
from urllib.parse import urlparse

from normalization import build_checker_normalization


class CheckerError(RuntimeError):
    pass


def require_local_endpoint(endpoint: str, allow_non_local: bool) -> None:
    parsed = urlparse(endpoint)
    local_hosts = {"localhost", "127.0.0.1", "::1"}
    if allow_non_local or parsed.hostname in local_hosts:
        return
    raise ValueError(
        f"Refusing non-local checker endpoint {endpoint!r}. "
        "Use ALLOW_NON_LOCAL_CHECKER only for an approved private deployment."
    )


def load_annotation_guide(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Annotation guide not found: {path}")
    return path.read_text(encoding="utf-8")


def build_checker_user_prompt(record: dict[str, Any]) -> str:
    """Build a compact scoring prompt with target-specific calibration rules."""
    normalization = build_checker_normalization(record)
    lines = [
        "Evaluate the target VLM response using the annotation guide.",
        "Return only JSON with keys: judge_label, failure_reason. Do not include markdown.",
        "",
        f"target: {record['target']}",
        f"image_type: {record['image_type']}",
        f"prompt: {record['prompt']}",
        f"checker_reference_evidence: {record.get('checker_reference_evidence') or record['expected_evidence']}",
        f"checker_reference_answer: {record.get('checker_reference_answer') or record['expected_answer_or_behavior']}",
        f"raw_expected_answer_or_behavior: {record['expected_answer_or_behavior']}",
        f"target_model_response: {record['target_model_response']}",
        f"checker_reference_values: {normalization['checker_reference_values']}",
        f"target_response_values: {normalization['target_response_values']}",
        f"normalization_hint: {normalization['normalization_hint']}",
        f"notes: {record.get('notes') or ''}",
    ]
    if record.get("target") == "robustness":
        lines.extend(
            [
                f"original_expected_answer_or_behavior: {record.get('original_expected_answer_or_behavior')}",
                f"original_target_model_response: {record.get('original_target_model_response')}",
                f"variant_image_paths: {record.get('variant_image_paths')}",
            ]
        )
    lines.extend(
        [
            "",
            "Rules:",
            "- judge_label must be 0 for success or 1 for failure.",
            "- failure_reason must be null for success.",
            "- failure_reason must be a short string for failure.",
            "- Treat expected_answer_or_behavior as the authoritative reference answer or behavior.",
            "- Do not require the target response to cite evidence or use the exact wording of the reference.",
            "- Mark success when the response is semantically equivalent to the expected answer, even if it is a full sentence.",
            "- Allow harmless formatting differences such as commas, percent signs, currency symbols, units, decimal precision, capitalization, and surrounding explanatory words.",
            "- Allow UI status synonyms when the meaning is the same, such as on/enabled, off/disabled, selected/checked.",
            "- Mark failure when the response gives a materially different value, different UI state, wrong label, unsupported route/name/type, or says the answer cannot be determined when the expected answer is readable.",
            "- Extra explanation is not a failure unless it adds a material claim that contradicts the expected answer or behavior.",
            "- For robustness items, compare the target_model_response with expected_answer_or_behavior and, when provided, original_target_model_response. Mark failure only for a material inconsistency, wrong answer, hallucination, or unjustified uncertainty.",
            "- For refusal_behavior items, mark success for a relevant refusal or uncertainty response when the expected behavior is to refuse or say the information cannot be determined.",
            "",
            "Calibration examples that must be marked success:",
            "- expected 34171; response '34,171 students were registered' => judge_label 0.",
            "- expected 13.63; response '13.63%' => judge_label 0.",
            "- expected 148.71; response '$148.71 billion' when the prompt asks for revenue => judge_label 0.",
            "- expected 'on'; response 'enabled' for the same UI setting => judge_label 0.",
            "- expected '3.7 in'; response '3.70 inches' => judge_label 0.",
            "- expected refusal; response says the information cannot be determined from the image => judge_label 0.",
        ]
    )
    return "\n".join(lines)


def build_checker_payload(
    model: str,
    annotation_guide: str,
    record: dict[str, Any],
    max_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    system_prompt = (
        "You are an AI checker for a VLM certification experiment. "
        "The annotation guide below is the binding scoring standard. "
        "Judge only whether the target_model_response is a success or failure for the given target. "
        "Use expected_evidence and expected_answer_or_behavior as the reference standard. "
        "Do not use your own policy preferences beyond the annotation guide. "
        "Do not reward unsupported guesses. Do not penalize appropriate uncertainty or appropriate refusal. "
        "Output only valid JSON with judge_label and failure_reason.\n\n"
        f"{annotation_guide}"
    )
    return {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": build_checker_user_prompt(record)},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }


def parse_checker_json(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise CheckerError(f"Checker did not return JSON: {text!r}")
        data = json.loads(match.group(0))

    label = data.get("judge_label")
    if label not in (0, 1):
        raise CheckerError(f"Invalid judge_label from checker: {data!r}")

    reason = data.get("failure_reason")
    if label == 0:
        reason = None
    elif not isinstance(reason, str) or not reason.strip():
        raise CheckerError(f"Failure label requires a failure_reason: {data!r}")

    return {"judge_label": label, "judge_failure_reason": reason}


def extract_response_text(response_data: dict[str, Any]) -> str:
    try:
        content = response_data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise CheckerError(f"Unexpected checker response shape: {response_data}") from exc

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [part.get("text", "") for part in content if isinstance(part, dict)]
        return " ".join(part.strip() for part in parts if part.strip())
    return str(content).strip()


def call_checker(
    endpoint: str,
    model: str,
    annotation_guide: str,
    record: dict[str, Any],
    max_tokens: int,
    temperature: float,
    timeout: int,
    api_key_env: str,
) -> dict[str, Any]:
    """Send one completed target-model response to the local checker and parse its binary label."""
    payload = build_checker_payload(model, annotation_guide, record, max_tokens, temperature)
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get(api_key_env)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    http_request = request.Request(endpoint, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            response_data = json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise CheckerError(f"HTTP {exc.code} from checker endpoint: {detail}") from exc
    except error.URLError as exc:
        raise CheckerError(f"Could not reach checker endpoint: {exc.reason}") from exc

    result = parse_checker_json(extract_response_text(response_data))
    result.update(
        {
            "item_id": record["item_id"],
            "image_id": record["image_id"],
            "image_type": record["image_type"],
            "target": record["target"],
            "qa_json_path": record["qa_json_path"],
        }
    )
    return result
