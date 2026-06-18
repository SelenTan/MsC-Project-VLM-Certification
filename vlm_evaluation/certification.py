#!/usr/bin/env python3
"""Certification calculations for imperfect-judge VLM evaluation."""

from __future__ import annotations

import math
import random
import statistics
from statistics import NormalDist
from typing import Any, Iterable


def label_int(value: Any, field_name: str) -> int:
    if value in (0, 1):
        return int(value)
    raise ValueError(f"{field_name} must be 0 or 1, got {value!r}")


def alpha_grid(alpha_min: float, alpha_max: float, alpha_step: float) -> list[float]:
    values: list[float] = []
    current = alpha_min
    while current <= alpha_max + 1e-12:
        values.append(round(current, 10))
        current += alpha_step
    return values


def estimate_reliability(records: Iterable[dict[str, Any]], target: str) -> dict[str, Any]:
    rows = [record for record in records if record["target"] == target]
    n_m1 = 0
    n_m0 = 0
    true_positive = 0
    false_positive = 0

    for record in rows:
        human = label_int(record.get("human_label"), "human_label")
        judge = label_int(record.get("judge_label"), "judge_label")
        if human == 1:
            n_m1 += 1
            if judge == 1:
                true_positive += 1
        else:
            n_m0 += 1
            if judge == 1:
                false_positive += 1

    tpr = true_positive / n_m1 if n_m1 else None
    fpr = false_positive / n_m0 if n_m0 else None
    reliable = tpr is not None and fpr is not None and tpr > fpr
    reason = None
    if tpr is None:
        reason = "No human-labelled failures in D_M, so TPR cannot be estimated."
    elif fpr is None:
        reason = "No human-labelled successes in D_M, so FPR cannot be estimated."
    elif tpr <= fpr:
        reason = "Checker is unreliable for this target because TPR <= FPR."

    return {
        "target": target,
        "n_records": len(rows),
        "n_M1": n_m1,
        "n_M0": n_m0,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "TPR": tpr,
        "FPR": fpr,
        "reliable": reliable,
        "unreliable_reason": reason,
    }


def certification_threshold(
    alpha: float,
    zeta: float,
    n_j: int,
    n_m1: int,
    n_m0: int,
    tpr: float,
    fpr: float,
) -> float:
    z_zeta = NormalDist().inv_cdf(zeta)
    alpha_prime = fpr + (tpr - fpr) * alpha
    variance = (
        alpha_prime * (1 - alpha_prime) / n_j
        + alpha**2 * tpr * (1 - tpr) / n_m1
        + (1 - alpha) ** 2 * fpr * (1 - fpr) / n_m0
    )
    return alpha_prime + z_zeta * math.sqrt(max(0.0, variance))


def certifiable_alpha(
    d_m: list[dict[str, Any]],
    d_j: list[dict[str, Any]],
    target: str,
    zeta: float,
    alpha_values: list[float],
) -> dict[str, Any]:
    """Estimate judge reliability on D_M, scan alpha, and return the first rejecting threshold."""
    reliability = estimate_reliability(d_m, target)
    if not reliability["reliable"]:
        return {
            "status": "invalid_reliability",
            "target": target,
            "certifiable_alpha": None,
            "R_J_hat": None,
            "c_J_prime": None,
            **reliability,
        }

    judge_labels = [label_int(record.get("judge_label"), "judge_label") for record in d_j]
    r_j_hat = sum(judge_labels) / len(judge_labels)
    last_threshold = None

    for alpha in alpha_values:
        threshold = certification_threshold(
            alpha=alpha,
            zeta=zeta,
            n_j=len(d_j),
            n_m1=reliability["n_M1"],
            n_m0=reliability["n_M0"],
            tpr=reliability["TPR"],
            fpr=reliability["FPR"],
        )
        last_threshold = threshold
        if r_j_hat < threshold:
            return {
                "status": "certified",
                "target": target,
                "certifiable_alpha": alpha,
                "R_J_hat": r_j_hat,
                "c_J_prime": threshold,
                **reliability,
            }

    return {
        "status": "not_certified",
        "target": target,
        "certifiable_alpha": None,
        "R_J_hat": r_j_hat,
        "c_J_prime": last_threshold,
        **reliability,
    }


def run_monte_carlo(
    human_gold_records: list[dict[str, Any]],
    evaluation_records: list[dict[str, Any]],
    targets: tuple[str, ...],
    n_m: int,
    n_j: int,
    repeats: int,
    zeta: float,
    alpha_min: float,
    alpha_max: float,
    alpha_step: float,
    seed: int,
) -> list[dict[str, Any]]:
    """Run the full repeated certification workflow with fixed-size D_M and D_J samples."""
    rng = random.Random(seed)
    alpha_values = alpha_grid(alpha_min, alpha_max, alpha_step)
    results: list[dict[str, Any]] = []

    for target in targets:
        gold = [record for record in human_gold_records if record["target"] == target]
        evaluation = [record for record in evaluation_records if record["target"] == target]
        if len(gold) < n_m:
            raise ValueError(f"Target {target} has {len(gold)} gold records, but N_M={n_m}.")
        if len(evaluation) < n_j:
            raise ValueError(f"Target {target} has {len(evaluation)} evaluation records, but N_J={n_j}.")

        for repeat_index in range(repeats):
            d_m = rng.sample(gold, n_m)
            d_j = rng.sample(evaluation, n_j)
            result = certifiable_alpha(d_m, d_j, target, zeta, alpha_values)
            result.update(
                {
                    "repeat_index": repeat_index,
                    "n_M": n_m,
                    "n_J": n_j,
                    "zeta": zeta,
                    "alpha_min": alpha_min,
                    "alpha_max": alpha_max,
                    "alpha_step": alpha_step,
                }
            )
            results.append(result)

    return results


def summarize_certificates(
    monte_carlo_results: list[dict[str, Any]],
    reliability_rows: list[dict[str, Any]],
    targets: tuple[str, ...],
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    reliability_by_target = {row["target"]: row for row in reliability_rows}

    for target in targets:
        rows = [row for row in monte_carlo_results if row["target"] == target]
        alphas = [row["certifiable_alpha"] for row in rows if row.get("certifiable_alpha") is not None]
        reliability = reliability_by_target[target]
        target_summary = {
            "target": target,
            "reliable_checker": reliability["reliable"],
            "unreliable_reason": reliability["unreliable_reason"],
            "true_positive": reliability["true_positive"],
            "false_positive": reliability["false_positive"],
            "TPR": reliability["TPR"],
            "FPR": reliability["FPR"],
            "gold_n_records": reliability["n_records"],
            "gold_n_M1": reliability["n_M1"],
            "gold_n_M0": reliability["n_M0"],
            "repeats": len(rows),
            "certified_repeats": len(alphas),
            "certified_rate": len(alphas) / len(rows) if rows else None,
            "alpha_mean": statistics.mean(alphas) if alphas else None,
            "alpha_median": statistics.median(alphas) if alphas else None,
            "alpha_min_observed": min(alphas) if alphas else None,
            "alpha_max_observed": max(alphas) if alphas else None,
            "status": (
                "checker_unreliable"
                if not reliability["reliable"]
                else "certified_in_some_repeats"
                if alphas
                else "not_certified"
            ),
        }
        summary.append(target_summary)

    return summary
