from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "vlm_evaluation"))

from error_experiments import (  # noqa: E402
    METHODS,
    PPI_VARIANTS,
    ppi_decision,
    run_calibration_stability,
    run_type_error_experiment,
)


def make_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(20):
        human = 1 if index < 6 else 0
        if human == 1:
            judge = 1 if index < 5 else 0
        else:
            judge = 1 if index in {6, 7} else 0
        records.append(
            {
                "record_key": f"key-{index}",
                "target": "visual_factuality",
                "human_label": human,
                "judge_label": judge,
            }
        )
    return records


class ErrorExperimentTest(unittest.TestCase):
    def test_type_error_experiment_includes_all_methods(self) -> None:
        rows = run_type_error_experiment(
            records=make_records(),
            targets=("visual_factuality",),
            alpha_values=[0.25, 0.50],
            n_m_values=(8,),
            n_j=10,
            repeats=5,
            zeta=0.05,
            seed=1,
            ridge_penalty=0.01,
        )

        self.assertEqual({row["method"] for row in rows}, set(METHODS))
        self.assertEqual({row["alpha"] for row in rows}, {0.25, 0.50})
        self.assertTrue(all(row["true_failure_rate"] == 0.3 for row in rows))

    def test_type_error_experiment_resumes_from_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "type_i_type_ii_checkpoint.csv"
            first_rows = run_type_error_experiment(
                records=make_records(),
                targets=("visual_factuality",),
                alpha_values=[0.25],
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=1,
                ridge_penalty=0.01,
                checkpoint_path=checkpoint_path,
                progress_interval=1,
            )
            second_rows = run_type_error_experiment(
                records=make_records(),
                targets=("visual_factuality",),
                alpha_values=[0.25],
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=1,
                ridge_penalty=0.01,
                checkpoint_path=checkpoint_path,
                progress_interval=1,
            )

            self.assertEqual(len(first_rows), len(METHODS))
            self.assertEqual(len(second_rows), len(METHODS))

    def test_calibration_stability_varies_over_n_m_grid(self) -> None:
        rows = run_calibration_stability(
            records=make_records(),
            targets=("visual_factuality",),
            n_m_values=(8, 12),
            repeats=5,
            seed=2,
        )

        self.assertEqual([row["n_M"] for row in rows], [8, 12])
        self.assertTrue(all(row["valid_repeats"] > 0 for row in rows))

    def test_ppi_variants_return_wald_decisions(self) -> None:
        records = make_records()
        d_m = records[:10]
        d_j = records[10:]

        for variant in PPI_VARIANTS:
            result = ppi_decision(d_m, d_j, alpha=0.5, zeta=0.05, variant=variant, ridge_penalty=0.01)
            self.assertIn(result["reject"], {True, False})
            self.assertIn("ppi_weight", result)


if __name__ == "__main__":
    unittest.main()
