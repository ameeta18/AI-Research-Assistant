"""Run the deterministic evaluation suite and enforce regression thresholds."""

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation_suite import (  # noqa: E402
    baseline_from_report,
    compare_with_baseline,
    load_evaluation_dataset,
    run_offline_evaluation,
    write_evaluation_report,
)


DEFAULT_DATASET = PROJECT_ROOT / "evals" / "dataset.json"
DEFAULT_BASELINE = PROJECT_ROOT / "evals" / "baseline.json"
DEFAULT_JSON_REPORT = PROJECT_ROOT / "output" / "evaluations" / "latest.json"
DEFAULT_MARKDOWN_REPORT = PROJECT_ROOT / "output" / "evaluations" / "latest.md"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run offline AI quality evaluation and regression checks."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--json-report", type=Path, default=DEFAULT_JSON_REPORT)
    parser.add_argument(
        "--markdown-report",
        type=Path,
        default=DEFAULT_MARKDOWN_REPORT,
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Replace the baseline with the current evaluated metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dataset = load_evaluation_dataset(args.dataset)
    report = run_offline_evaluation(dataset)

    if args.update_baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(
            json.dumps(baseline_from_report(report), indent=2, sort_keys=True),
            encoding="utf-8",
        )

    if not args.baseline.exists():
        print(
            f"Baseline not found: {args.baseline}. "
            "Run once with --update-baseline after reviewing the metrics.",
            file=sys.stderr,
        )
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    report["regression"] = compare_with_baseline(
        report,
        baseline,
        dataset.thresholds,
    )
    write_evaluation_report(report, args.json_report, args.markdown_report)

    status = "PASS" if report["regression"]["passed"] else "FAIL"
    print(f"Evaluation regression status: {status}")
    for name, comparison in report["regression"]["metrics"].items():
        print(
            f"  {name}: {comparison['current']} "
            f"(baseline {comparison['baseline']}, "
            f"delta {comparison['delta']})"
        )
    print(f"JSON report: {args.json_report}")
    print(f"Markdown report: {args.markdown_report}")
    return 0 if report["regression"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
