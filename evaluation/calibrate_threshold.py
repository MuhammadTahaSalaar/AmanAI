"""Calibrate MIN_RERANK_SCORE by sweeping over [0.1 .. 0.6].

Runs each question in the golden set through the retrieval pipeline only
(no generation) and records the top rerank score.  Then, for each candidate
threshold value, it reports how well that threshold separates in-domain
questions (should pass) from out-of-domain questions (should be refused).

Usage (from repo root):
    python -m evaluation.calibrate_threshold

The script prints a table and recommends the threshold that maximises the
separation gap (in_domain pass-rate minus out_of_domain pass-rate).

AWS credentials + SUPABASE_* env vars must be set before running.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

GOLDEN_PATH = Path(__file__).parent / "golden_dataset_v2.json"
SWEEP_VALUES = [round(v * 0.1, 1) for v in range(1, 7)]  # 0.1 … 0.6


def _get_top_score(question: str) -> float:
    """Return the top rerank score for a question without generating."""
    try:
        from backend.app.retrieval import retrieve

        _docs, top_score = retrieve(question)
        return top_score
    except Exception as exc:
        logger.error("retrieve() failed for %r: %s", question[:60], exc)
        return 0.0


def run_calibration() -> None:
    with open(GOLDEN_PATH, "r", encoding="utf-8") as fh:
        golden = json.load(fh)

    logger.info("Loaded %d samples; collecting top rerank scores …", len(golden))

    in_domain_scores: list[float] = []
    out_domain_scores: list[float] = []

    for i, sample in enumerate(golden):
        question = sample["question"]
        domain = sample.get("domain", "in_domain")
        logger.info("[%d/%d] %-10s  %s", i + 1, len(golden), domain, question[:70])

        score = _get_top_score(question)

        if domain == "out_of_domain":
            out_domain_scores.append(score)
        else:
            in_domain_scores.append(score)

    if not in_domain_scores or not out_domain_scores:
        logger.error(
            "Need at least one in-domain AND one out-of-domain sample. "
            "Check the 'domain' field in %s.",
            GOLDEN_PATH,
        )
        return

    logger.info(
        "In-domain: %d samples | Out-of-domain: %d samples",
        len(in_domain_scores),
        len(out_domain_scores),
    )

    # ------------------------------------------------------------------
    # Sweep thresholds
    # ------------------------------------------------------------------
    header = (
        f"{'Threshold':>10}  "
        f"{'InDom Pass%':>12}  "
        f"{'OutDom Pass%':>13}  "
        f"{'Gap':>8}"
    )
    logger.info(header)
    logger.info("-" * len(header))

    best_threshold = SWEEP_VALUES[0]
    best_gap = -1.0
    table: list[dict] = []

    for threshold in SWEEP_VALUES:
        in_pass = sum(1 for s in in_domain_scores if s >= threshold) / len(
            in_domain_scores
        )
        out_pass = sum(1 for s in out_domain_scores if s >= threshold) / len(
            out_domain_scores
        )
        gap = in_pass - out_pass

        row = {
            "threshold": threshold,
            "in_domain_pass_rate": round(in_pass, 4),
            "out_domain_pass_rate": round(out_pass, 4),
            "gap": round(gap, 4),
        }
        table.append(row)
        logger.info(
            "%10.1f  %12.1f%%  %13.1f%%  %8.4f",
            threshold,
            in_pass * 100,
            out_pass * 100,
            gap,
        )

        if gap > best_gap:
            best_gap = gap
            best_threshold = threshold

    logger.info("")
    logger.info(
        "Recommended MIN_RERANK_SCORE = %.1f  (gap = %.4f)",
        best_threshold,
        best_gap,
    )
    logger.info(
        "Set this in your .env or as the MIN_RERANK_SCORE environment variable."
    )

    # Save detailed results alongside the evaluation outputs
    report_path = Path(__file__).parent / "threshold_calibration.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "sweep": table,
                "recommended": best_threshold,
                "gap": best_gap,
                "in_domain_scores": in_domain_scores,
                "out_domain_scores": out_domain_scores,
            },
            fh,
            indent=2,
        )
    logger.info("Full calibration data saved to %s", report_path)


if __name__ == "__main__":
    run_calibration()
