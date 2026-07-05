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
    combined_type_error_value,
    failure_heatmap_rows,
    fixed_alpha_score_rows,
    make_error_experiment_charts,
    observed_judge_points,
    paper_alpha_xlim,
    ppi_decision,
    run_calibration_stability,
    run_noisy_type_ii_by_n_m,
    run_type_error_experiment,
    write_error_experiment_artifacts,
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

    def test_noisy_type_ii_sweep_varies_over_n_m(self) -> None:
        rows = run_noisy_type_ii_by_n_m(
            records=make_records(),
            targets=("visual_factuality",),
            n_m_values=(8, 12),
            n_j=10,
            repeats=5,
            zeta=0.05,
            seed=3,
            alpha_margin=0.10,
        )

        self.assertEqual([row["n_M"] for row in rows], [8, 12])
        self.assertTrue(all(row["method"] == "noisy_ht" for row in rows))
        self.assertTrue(all(row["type_ii_error"] is not None for row in rows))

    def test_ppi_variants_return_wald_decisions(self) -> None:
        records = make_records()
        d_m = records[:10]
        d_j = records[10:]

        for variant in PPI_VARIANTS:
            result = ppi_decision(d_m, d_j, alpha=0.5, zeta=0.05, variant=variant, ridge_penalty=0.01)
            self.assertIn(result["reject"], {True, False})
            self.assertIn("ppi_weight", result)

    def test_combined_plot_uses_type_i_left_and_type_ii_right_of_true_rate(self) -> None:
        left = {"alpha": 0.2, "true_failure_rate": 0.3, "type_i_error": 0.04, "type_ii_error": None}
        right = {"alpha": 0.4, "true_failure_rate": 0.3, "type_i_error": None, "type_ii_error": 0.7}

        self.assertEqual(combined_type_error_value(left), 0.04)
        self.assertEqual(combined_type_error_value(right), 0.7)

    def test_paper_alpha_xlim_keeps_padding_but_drops_empty_right_tail(self) -> None:
        rows = [
            {"alpha": 0.20, "true_failure_rate": 0.30, "type_i_error": 0.02, "type_ii_error": None},
            {"alpha": 0.30, "true_failure_rate": 0.30, "type_i_error": 0.03, "type_ii_error": None},
            {"alpha": 0.40, "true_failure_rate": 0.30, "type_i_error": None, "type_ii_error": 0.20},
            {"alpha": 0.50, "true_failure_rate": 0.30, "type_i_error": None, "type_ii_error": 0.00},
            {"alpha": 0.80, "true_failure_rate": 0.30, "type_i_error": None, "type_ii_error": 0.00},
        ]

        left, right = paper_alpha_xlim(rows)

        self.assertLess(left, 0.20)
        self.assertGreater(right, 0.40)
        self.assertLess(right, 0.80)
        true_rate_fraction = (0.30 - left) / (right - left)
        self.assertGreaterEqual(true_rate_fraction, 0.30)

    def test_observed_judge_points_use_main_n_m(self) -> None:
        rows = [
            {"target": "visual_factuality", "n_M": 25, "oracle_TPR": 0.8, "oracle_FPR": 0.2},
            {"target": "visual_factuality", "n_M": 50, "oracle_TPR": 0.9, "oracle_FPR": 0.1},
            {"target": "robustness", "n_M": 50, "oracle_TPR": 0.7, "oracle_FPR": 0.3},
        ]

        points = observed_judge_points(rows, main_n_m=50)

        self.assertEqual(
            points,
            [
                {"target": "robustness", "TPR": 0.7, "FPR": 0.3},
                {"target": "visual_factuality", "TPR": 0.9, "FPR": 0.1},
            ],
        )

    def test_failure_heatmap_rows_group_by_target_and_image_type(self) -> None:
        rows = failure_heatmap_rows(
            [
                {"target": "visual_factuality", "image_type": "forms", "human_label": "1", "label_status": "complete"},
                {"target": "visual_factuality", "image_type": "forms", "human_label": "0", "label_status": "complete"},
                {"target": "visual_factuality", "image_type": "charts", "human_label": "0", "label_status": "complete"},
                {"target": "robustness", "image_type": "forms", "human_label": "1", "label_status": "draft"},
            ]
        )

        self.assertEqual(
            rows,
            [
                {"target": "visual_factuality", "image_type": "charts", "n": 1, "failure_rate": 0.0},
                {"target": "visual_factuality", "image_type": "forms", "n": 2, "failure_rate": 0.5},
            ],
        )

    def test_fixed_alpha_score_rows_include_overall_and_target_scores(self) -> None:
        rows = fixed_alpha_score_rows(
            [
                {"target": "visual_factuality", "image_type": "forms", "human_label": "1", "label_status": "complete"},
                {"target": "visual_factuality", "image_type": "forms", "human_label": "0", "label_status": "complete"},
                {"target": "robustness", "image_type": "charts", "human_label": "0", "label_status": "complete"},
            ],
            fixed_alpha=0.25,
        )

        overall = next(row for row in rows if row["target"] == "overall")
        visual = next(row for row in rows if row["target"] == "visual_factuality" and row["image_type"] == "all")
        self.assertAlmostEqual(overall["score"], 100 * (2 / 3))
        self.assertEqual(visual["score"], 50.0)
        self.assertFalse(visual["passes_fixed_alpha"])

    def test_paper_style_charts_write_one_combined_figure_per_target(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            rows = run_type_error_experiment(
                records=make_records(),
                targets=("visual_factuality",),
                alpha_values=[0.25, 0.50],
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=1,
                ridge_penalty=0.01,
            )
            type_ii_rows = run_noisy_type_ii_by_n_m(
                records=make_records(),
                targets=("visual_factuality",),
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=3,
                alpha_margin=0.10,
            )
            calibration_rows = run_calibration_stability(
                records=make_records(),
                targets=("visual_factuality",),
                n_m_values=(8,),
                repeats=3,
                seed=2,
            )

            try:
                make_error_experiment_charts(Path(temp_dir), rows, calibration_rows, type_ii_rows, main_n_m=8, main_n_j=10, fixed_alpha=0.25)
            except RuntimeError as exc:
                self.skipTest(str(exc))

            self.assertTrue((Path(temp_dir) / "type_i_type_ii_by_alpha_visual_factuality.png").exists())
            self.assertTrue((Path(temp_dir) / "calibration_stability_by_n_m_visual_factuality.png").exists())
            self.assertTrue((Path(temp_dir) / "type_ii_error_by_n_m.png").exists())

    def test_error_artifacts_keep_final_csv_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            rows = run_type_error_experiment(
                records=make_records(),
                targets=("visual_factuality",),
                alpha_values=[0.25],
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=1,
                ridge_penalty=0.01,
            )
            type_ii_rows = run_noisy_type_ii_by_n_m(
                records=make_records(),
                targets=("visual_factuality",),
                n_m_values=(8,),
                n_j=10,
                repeats=3,
                zeta=0.05,
                seed=3,
                alpha_margin=0.10,
            )
            calibration_rows = run_calibration_stability(
                records=make_records(),
                targets=("visual_factuality",),
                n_m_values=(8,),
                repeats=3,
                seed=2,
            )
            try:
                write_error_experiment_artifacts(output_dir, rows, calibration_rows, type_ii_rows, 8, 10, 0.25)
            except RuntimeError as exc:
                self.skipTest(str(exc))

            self.assertTrue((output_dir / "type_i_type_ii_by_alpha.csv").exists())
            self.assertTrue((output_dir / "type_ii_by_n_m.csv").exists())
            self.assertTrue((output_dir / "calibration_stability.csv").exists())


if __name__ == "__main__":
    unittest.main()
