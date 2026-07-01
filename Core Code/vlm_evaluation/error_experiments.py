#!/usr/bin/env python3
"""Paper-style Type I/II error experiments for Noisy but Valid baselines."""

from __future__ import annotations

import math
import csv
import random
import statistics
from pathlib import Path
from statistics import NormalDist
from typing import Any, Iterable

from artifacts import write_csv
from certification import (
    alpha_grid,
    certification_threshold,
    estimate_reliability,
    label_int,
)


PPI_VARIANTS = ("ppi", "ppi_plus", "ridge_ppi")
METHODS = ("direct_ht", "noisy_ht", "oracle_noisy_ht", *PPI_VARIANTS)
PLOTTED_METHODS = METHODS
METHOD_LABELS = {
    "direct_ht": "Direct HT",
    "noisy_ht": "Noisy HT (general)",
    "oracle_noisy_ht": "Noisy HT (oracle)",
    "ppi": "PPI",
    "ppi_plus": "PPI++",
    "ridge_ppi": "Ridge PPI",
}
METHOD_STYLES = {
    "direct_ht": {"color": "#1f77b4", "linestyle": "-"},
    "noisy_ht": {"color": "#ff7f0e", "linestyle": "-"},
    "oracle_noisy_ht": {"color": "#2ca02c", "linestyle": "--"},
    "ppi": {"color": "#d62728", "linestyle": "-"},
    "ppi_plus": {"color": "#9467bd", "linestyle": "-"},
    "ridge_ppi": {"color": "#8c564b", "linestyle": "-"},
}


def mean(values: Iterable[float]) -> float:
    data = list(values)
    if not data:
        raise ValueError("mean requires at least one value.")
    return sum(data) / len(data)


def experiment_key(row: dict[str, Any]) -> tuple[str, int, int, int, str, str]:
    return (
        str(row["target"]),
        int(row["n_M"]),
        int(row["n_J"]),
        int(row["repeats"]),
        f"{float(row['zeta']):.12g}",
        f"{float(row['alpha']):.12g}",
    )


def coerce_optional_float(value: str) -> float | None:
    if value in ("", "None", None):
        return None
    return float(value)


def read_type_error_checkpoint(path: Path) -> list[dict[str, Any]]:
    """Load completed Type I/II rows so interrupted paper experiments can resume."""
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    for row in rows:
        row["alpha"] = float(row["alpha"])
        row["n_M"] = int(row["n_M"])
        row["n_J"] = int(row["n_J"])
        row["repeats"] = int(row["repeats"])
        row["zeta"] = float(row["zeta"])
        row["true_failure_rate"] = float(row["true_failure_rate"])
        row["oracle_TPR"] = coerce_optional_float(row["oracle_TPR"])
        row["oracle_FPR"] = coerce_optional_float(row["oracle_FPR"])
        row["type_i_error"] = coerce_optional_float(row["type_i_error"])
        row["type_ii_error"] = coerce_optional_float(row["type_ii_error"])
        row["power"] = coerce_optional_float(row["power"])
    return rows


def completed_type_error_keys(rows: list[dict[str, Any]]) -> set[tuple[str, int, int, int, str, str]]:
    methods_by_key: dict[tuple[str, int, int, int, str, str], set[str]] = {}
    for row in rows:
        methods_by_key.setdefault(experiment_key(row), set()).add(str(row["method"]))
    return {key for key, methods in methods_by_key.items() if methods == set(METHODS)}


def sample_variance(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return statistics.variance(values)


def covariance(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("covariance inputs must have the same length.")
    if len(left) < 2:
        return 0.0
    left_mean = mean(left)
    right_mean = mean(right)
    return sum((x - left_mean) * (y - right_mean) for x, y in zip(left, right)) / (len(left) - 1)


def direct_ht_decision(d_m: list[dict[str, Any]], alpha: float, zeta: float) -> dict[str, Any]:
    labels = [label_int(record.get("human_label"), "human_label") for record in d_m]
    r_m_hat = mean(labels)
    threshold = alpha + NormalDist().inv_cdf(zeta) * math.sqrt(max(0.0, alpha * (1 - alpha) / len(labels)))
    return {
        "reject": r_m_hat < threshold,
        "statistic": r_m_hat,
        "threshold": threshold,
    }


def noisy_ht_decision(d_m: list[dict[str, Any]], d_j: list[dict[str, Any]], alpha: float, zeta: float) -> dict[str, Any]:
    """Run the variance-corrected Noisy HT certification decision using estimated TPR/FPR."""
    reliability = estimate_reliability(d_m, d_m[0]["target"])
    if not reliability["reliable"]:
        return {
            "reject": False,
            "statistic": None,
            "threshold": None,
            **reliability,
        }
    judge_labels = [label_int(record.get("judge_label"), "judge_label") for record in d_j]
    r_j_hat = mean(judge_labels)
    threshold = certification_threshold(
        alpha=alpha,
        zeta=zeta,
        n_j=len(d_j),
        n_m1=reliability["n_M1"],
        n_m0=reliability["n_M0"],
        tpr=reliability["TPR"],
        fpr=reliability["FPR"],
    )
    return {
        "reject": r_j_hat < threshold,
        "statistic": r_j_hat,
        "threshold": threshold,
        **reliability,
    }


def oracle_parameters(records: list[dict[str, Any]], target: str) -> dict[str, Any]:
    reliability = estimate_reliability(records, target)
    if reliability["TPR"] is None or reliability["FPR"] is None:
        raise ValueError(f"Cannot estimate oracle TPR/FPR for {target}.")
    return reliability


def oracle_noisy_ht_decision(
    d_j: list[dict[str, Any]],
    alpha: float,
    zeta: float,
    tpr: float,
    fpr: float,
) -> dict[str, Any]:
    judge_labels = [label_int(record.get("judge_label"), "judge_label") for record in d_j]
    r_j_hat = mean(judge_labels)
    alpha_prime = fpr + (tpr - fpr) * alpha
    threshold = alpha_prime + NormalDist().inv_cdf(zeta) * math.sqrt(
        max(0.0, alpha_prime * (1 - alpha_prime) / len(judge_labels))
    )
    return {
        "reject": r_j_hat < threshold,
        "statistic": r_j_hat,
        "threshold": threshold,
        "TPR": tpr,
        "FPR": fpr,
    }


def ppi_weight(
    variant: str,
    human_labels: list[float],
    judge_dm: list[float],
    judge_dj: list[float],
    ridge_penalty: float,
) -> float:
    if variant == "ppi":
        return 1.0
    var_j_dm = sample_variance(judge_dm)
    var_j_dj = sample_variance(judge_dj)
    denominator = var_j_dm + (len(judge_dm) / len(judge_dj)) * var_j_dj
    if variant == "ridge_ppi":
        denominator += ridge_penalty
    if denominator <= 0:
        return 0.0
    return covariance(human_labels, judge_dm) / denominator


def ppi_decision(
    d_m: list[dict[str, Any]],
    d_j: list[dict[str, Any]],
    alpha: float,
    zeta: float,
    variant: str,
    ridge_penalty: float,
) -> dict[str, Any]:
    """Run a PPI-family Wald test that combines human labels with judge-labelled records."""
    if variant not in PPI_VARIANTS:
        raise ValueError(f"Unknown PPI variant: {variant}")
    human = [float(label_int(record.get("human_label"), "human_label")) for record in d_m]
    judge_dm = [float(label_int(record.get("judge_label"), "judge_label")) for record in d_m]
    judge_dj = [float(label_int(record.get("judge_label"), "judge_label")) for record in d_j]
    weight = ppi_weight(variant, human, judge_dm, judge_dj, ridge_penalty)
    statistic = mean(human) + weight * (mean(judge_dj) - mean(judge_dm))
    variance = sample_variance([y - weight * j for y, j in zip(human, judge_dm)]) / len(d_m)
    variance += (weight**2) * sample_variance(judge_dj) / len(d_j)
    threshold = alpha + NormalDist().inv_cdf(zeta) * math.sqrt(max(0.0, variance))
    return {
        "reject": statistic < threshold,
        "statistic": statistic,
        "threshold": threshold,
        "ppi_weight": weight,
    }


def method_decision(
    method: str,
    d_m: list[dict[str, Any]],
    d_j: list[dict[str, Any]],
    alpha: float,
    zeta: float,
    oracle: dict[str, Any],
    ridge_penalty: float,
) -> dict[str, Any]:
    if method == "direct_ht":
        return direct_ht_decision(d_m, alpha, zeta)
    if method == "noisy_ht":
        return noisy_ht_decision(d_m, d_j, alpha, zeta)
    if method == "oracle_noisy_ht":
        return oracle_noisy_ht_decision(d_j, alpha, zeta, oracle["TPR"], oracle["FPR"])
    if method in PPI_VARIANTS:
        return ppi_decision(d_m, d_j, alpha, zeta, method, ridge_penalty)
    raise ValueError(f"Unknown method: {method}")


def run_type_error_experiment(
    records: list[dict[str, Any]],
    targets: tuple[str, ...],
    alpha_values: list[float],
    n_m_values: tuple[int, ...],
    n_j: int,
    repeats: int,
    zeta: float,
    seed: int,
    ridge_penalty: float,
    checkpoint_path: Path | None = None,
    progress_interval: int = 10,
) -> list[dict[str, Any]]:
    """Estimate paper-style Type I/II rates for each method, target, alpha, and n_M."""
    rows: list[dict[str, Any]] = read_type_error_checkpoint(checkpoint_path) if checkpoint_path else []
    completed_keys = completed_type_error_keys(rows)
    progress_interval = max(1, progress_interval)
    total_combinations = len(targets) * len(n_m_values) * len(alpha_values)
    combination_index = 0
    for target in targets:
        target_records = [record for record in records if record["target"] == target]
        if len(target_records) < n_j:
            raise ValueError(f"Target {target} has {len(target_records)} records, but n_J={n_j}.")
        true_failure_rate = mean(label_int(record.get("human_label"), "human_label") for record in target_records)
        oracle = oracle_parameters(target_records, target)
        for n_m in n_m_values:
            if len(target_records) < n_m:
                raise ValueError(f"Target {target} has {len(target_records)} records, but n_M={n_m}.")
            for alpha in alpha_values:
                combination_index += 1
                key = (target, n_m, n_j, repeats, f"{float(zeta):.12g}", f"{float(alpha):.12g}")
                if key in completed_keys:
                    continue
                rng = random.Random(f"{seed}:{target}:{n_m}:{alpha:.12g}")
                counts = {
                    method: {"type_i": 0, "type_ii": 0, "eligible_i": 0, "eligible_ii": 0}
                    for method in METHODS
                }
                for repeat_index in range(repeats):
                    d_m = rng.sample(target_records, n_m)
                    d_j = rng.sample(target_records, n_j)
                    h0_true = true_failure_rate >= alpha
                    for method in METHODS:
                        decision = method_decision(method, d_m, d_j, alpha, zeta, oracle, ridge_penalty)
                        if h0_true:
                            counts[method]["eligible_i"] += 1
                            if decision["reject"]:
                                counts[method]["type_i"] += 1
                        else:
                            counts[method]["eligible_ii"] += 1
                            if not decision["reject"]:
                                counts[method]["type_ii"] += 1
                for method, method_counts in counts.items():
                    eligible_i = method_counts["eligible_i"]
                    eligible_ii = method_counts["eligible_ii"]
                    rows.append(
                        {
                            "target": target,
                            "method": method,
                            "alpha": alpha,
                            "n_M": n_m,
                            "n_J": n_j,
                            "repeats": repeats,
                            "zeta": zeta,
                            "true_failure_rate": true_failure_rate,
                            "oracle_TPR": oracle["TPR"],
                            "oracle_FPR": oracle["FPR"],
                            "type_i_error": method_counts["type_i"] / eligible_i if eligible_i else None,
                            "type_ii_error": method_counts["type_ii"] / eligible_ii if eligible_ii else None,
                            "power": 1 - method_counts["type_ii"] / eligible_ii if eligible_ii else None,
                        }
                    )
                if checkpoint_path is not None:
                    write_csv(checkpoint_path, rows)
                if combination_index == 1 or combination_index == total_combinations or combination_index % progress_interval == 0:
                    print(
                        f"Type I/II experiments {combination_index}/{total_combinations}: "
                        f"{target}, n_M={n_m}, alpha={alpha:.2f}"
                    )
    return rows


def run_calibration_stability(
    records: list[dict[str, Any]],
    targets: tuple[str, ...],
    n_m_values: tuple[int, ...],
    repeats: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Estimate how TPR/FPR estimates vary as the calibration size n_M changes."""
    rng = random.Random(seed)
    rows: list[dict[str, Any]] = []
    for target in targets:
        target_records = [record for record in records if record["target"] == target]
        for n_m in n_m_values:
            tprs: list[float] = []
            fprs: list[float] = []
            invalid = 0
            for _ in range(repeats):
                d_m = rng.sample(target_records, n_m)
                reliability = estimate_reliability(d_m, target)
                if reliability["TPR"] is None or reliability["FPR"] is None:
                    invalid += 1
                    continue
                tprs.append(reliability["TPR"])
                fprs.append(reliability["FPR"])
            rows.append(
                {
                    "target": target,
                    "n_M": n_m,
                    "repeats": repeats,
                    "valid_repeats": len(tprs),
                    "invalid_repeats": invalid,
                    "TPR_mean": statistics.mean(tprs) if tprs else None,
                    "TPR_stderr": statistics.stdev(tprs) / math.sqrt(len(tprs)) if len(tprs) > 1 else 0.0,
                    "FPR_mean": statistics.mean(fprs) if fprs else None,
                    "FPR_stderr": statistics.stdev(fprs) / math.sqrt(len(fprs)) if len(fprs) > 1 else 0.0,
                }
            )
    return rows


def make_error_experiment_charts(
    output_dir: Path,
    type_error_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    main_n_m: int,
    main_n_j: int,
) -> None:
    """Render compact paper-style Type I/II curves and the TPR/FPR decision region."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("Chart generation requires Matplotlib.") from exc

    for obsolete_name in ("type_i_error_by_alpha.png", "type_ii_error_by_alpha.png"):
        (output_dir / obsolete_name).unlink(missing_ok=True)

    targets = sorted({row["target"] for row in type_error_rows})
    for target in targets:
        n_m_values = sorted({int(row["n_M"]) for row in type_error_rows if row["target"] == target})
        fig, axes = plt.subplots(1, len(n_m_values), figsize=(4.9 * len(n_m_values), 3.9), sharey=True)
        if len(n_m_values) == 1:
            axes = [axes]
        for axis, n_m in zip(axes, n_m_values):
            panel_rows = [row for row in type_error_rows if row["target"] == target and int(row["n_M"]) == n_m]
            if not panel_rows:
                continue
            zeta = float(panel_rows[0]["zeta"])
            n_j = int(panel_rows[0]["n_J"])
            true_failure_rate = float(panel_rows[0]["true_failure_rate"])
            oracle_tpr = panel_rows[0].get("oracle_TPR")
            oracle_fpr = panel_rows[0].get("oracle_FPR")
            add_paper_background(axis)
            for method in PLOTTED_METHODS:
                rows = sorted(
                    [
                        row
                        for row in panel_rows
                        if row["method"] == method
                        and combined_type_error_value(row) is not None
                    ],
                    key=lambda row: row["alpha"],
                )
                if not rows:
                    continue
                axis.plot(
                    [row["alpha"] for row in rows],
                    [combined_type_error_value(row) for row in rows],
                    label=METHOD_LABELS[method],
                    linewidth=1.6,
                    **METHOD_STYLES[method],
                )
            axis.axhline(zeta, color="0.45", linestyle=":", linewidth=0.9)
            axis.axvline(true_failure_rate, color="0.25", linestyle="--", linewidth=0.9)
            axis.text(0.02, zeta + 0.025, rf"$\zeta={zeta:.2f}$", transform=axis.get_yaxis_transform(), fontsize=8)
            axis.text(
                true_failure_rate - 0.01,
                0.47,
                rf"$R_M={true_failure_rate:.2f}$",
                rotation=90,
                ha="right",
                va="center",
                fontsize=8,
            )
            axis.text(0.02, 0.86, "Type I Error Probability", transform=axis.transAxes, fontsize=9)
            axis.text(0.56, 0.86, "Type II Error Probability", transform=axis.transAxes, fontsize=9)
            axis.set_ylim(0.0, 1.0)
            axis.set_xlim(*paper_alpha_xlim(panel_rows))
            axis.set_xlabel(r"Threshold $\alpha$")
            tpr_text = "nan" if oracle_tpr is None else f"{float(oracle_tpr):.3f}"
            fpr_text = "nan" if oracle_fpr is None else f"{float(oracle_fpr):.3f}"
            axis.set_title(rf"$n_M$ = {n_m}, $n_J$ = {n_j}, TPR = {tpr_text}, FPR = {fpr_text}", fontsize=10)
        axes[0].set_ylabel(r"$P_e^{(I)} / P_e^{(II)}$")
        handles, labels = axes[-1].get_legend_handles_labels()
        fig.suptitle(target, y=1.03, fontsize=12, fontweight="bold")
        fig.legend(handles, labels, loc="lower center", ncol=min(6, len(labels)), fontsize=8)
        fig.tight_layout(rect=[0, 0.14, 1, 0.98])
        fig.savefig(output_dir / f"type_i_type_ii_by_alpha_{target}.png", dpi=180)
        plt.close()

    region_rows = decision_region_rows(alpha=0.25, zeta=0.05, n_m=main_n_m, n_j=main_n_j, true_failure_rate=0.10)
    tpr_values = sorted({row["TPR"] for row in region_rows})
    fpr_values = sorted({row["FPR"] for row in region_rows})
    grid = []
    for tpr in tpr_values:
        grid.append(
            [
                next(
                    row["noisy_advantage"]
                    for row in region_rows
                    if row["TPR"] == tpr and row["FPR"] == fpr
                )
                for fpr in fpr_values
            ]
        )
    plt.figure(figsize=(7, 5))
    max_abs = max(abs(value) for row in grid for value in row) or 1.0
    plt.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(fpr_values), max(fpr_values), min(tpr_values), max(tpr_values)],
        cmap="coolwarm",
        vmin=-max_abs,
        vmax=max_abs,
    )
    plt.colorbar(label="Direct Type II - Noisy Type II")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Noisy HT relative to Direct HT")
    plt.tight_layout()
    plt.savefig(output_dir / "tpr_fpr_decision_region.png", dpi=160)
    plt.close()

    if calibration_rows:
        make_calibration_stability_chart(output_dir, calibration_rows)


def add_paper_background(axis: Any) -> None:
    axis.patch.set_facecolor("white")
    axis.patch.set_hatch("//////")
    axis.patch.set_edgecolor("0.92")
    axis.grid(False)


def combined_type_error_value(row: dict[str, Any]) -> float | None:
    true_failure_rate = float(row["true_failure_rate"])
    alpha = float(row["alpha"])
    if alpha <= true_failure_rate:
        return row.get("type_i_error")
    return row.get("type_ii_error")


def paper_alpha_xlim(rows: list[dict[str, Any]]) -> tuple[float, float]:
    alphas = [float(row["alpha"]) for row in rows]
    if not alphas:
        return 0.0, 1.0
    left = min(alphas)
    right = max(alphas)
    padding = max((right - left) * 0.035, 0.01)
    return max(0.0, left - padding), min(1.0, right + padding)


def make_calibration_stability_chart(output_dir: Path, calibration_rows: list[dict[str, Any]]) -> None:
    targets = sorted({row["target"] for row in calibration_rows})
    fig, axes = plt.subplots(1, len(targets), figsize=(4.8 * len(targets), 3.4), sharey=True)
    if len(targets) == 1:
        axes = [axes]
    for axis, target in zip(axes, targets):
        rows = sorted([row for row in calibration_rows if row["target"] == target], key=lambda row: row["n_M"])
        n_m_values = [row["n_M"] for row in rows]
        axis.errorbar(
            n_m_values,
            [row["TPR_mean"] for row in rows],
            yerr=[row["TPR_stderr"] for row in rows],
            marker="o",
            linewidth=1.6,
            label="TPR",
        )
        axis.errorbar(
            n_m_values,
            [row["FPR_mean"] for row in rows],
            yerr=[row["FPR_stderr"] for row in rows],
            marker="o",
            linewidth=1.6,
            label="FPR",
        )
        axis.set_title(target)
        axis.set_xlabel(r"$n_M$")
        axis.set_ylim(0.0, 1.0)
    axes[0].set_ylabel("Estimated rate")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8)
    fig.tight_layout(rect=[0, 0.14, 1, 1])
    fig.savefig(output_dir / "calibration_stability_by_n_m.png", dpi=180)
    plt.close()


def normal_type_ii_error(true_rate: float, threshold: float, sample_size: int) -> float:
    variance = max(1e-12, true_rate * (1 - true_rate) / sample_size)
    z_value = (threshold - true_rate) / math.sqrt(variance)
    return 1 - NormalDist().cdf(z_value)


def decision_region_rows(
    alpha: float,
    zeta: float,
    n_m: int,
    n_j: int,
    true_failure_rate: float,
) -> list[dict[str, Any]]:
    """Approximate the TPR/FPR plane where oracle noisy testing beats direct testing."""
    direct_threshold = alpha + NormalDist().inv_cdf(zeta) * math.sqrt(max(0.0, alpha * (1 - alpha) / n_m))
    direct_type_ii = normal_type_ii_error(true_failure_rate, direct_threshold, n_m)
    rows: list[dict[str, Any]] = []
    for tpr_index in range(0, 21):
        tpr = round(tpr_index / 20, 2)
        for fpr_index in range(0, 21):
            fpr = round(fpr_index / 20, 2)
            if tpr <= fpr:
                noisy_type_ii = 1.0
            else:
                alpha_prime = fpr + (tpr - fpr) * alpha
                true_judge_rate = fpr + (tpr - fpr) * true_failure_rate
                threshold = alpha_prime + NormalDist().inv_cdf(zeta) * math.sqrt(
                    max(0.0, alpha_prime * (1 - alpha_prime) / n_j)
                )
                noisy_type_ii = normal_type_ii_error(true_judge_rate, threshold, n_j)
            rows.append(
                {
                    "alpha": alpha,
                    "true_failure_rate": true_failure_rate,
                    "n_M": n_m,
                    "n_J": n_j,
                    "TPR": tpr,
                    "FPR": fpr,
                    "direct_type_ii": direct_type_ii,
                    "oracle_noisy_type_ii": noisy_type_ii,
                    "noisy_advantage": direct_type_ii - noisy_type_ii,
                }
            )
    return rows


def write_error_experiment_artifacts(
    output_dir: Path,
    type_error_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    main_n_m: int,
    main_n_j: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    make_error_experiment_charts(output_dir / "charts", type_error_rows, calibration_rows, main_n_m, main_n_j)
    checkpoint_path = output_dir / "type_i_type_ii_checkpoint.csv"
    checkpoint_path.unlink(missing_ok=True)
