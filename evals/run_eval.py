import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.domain.models import Listing
from app.services.analysis import analyze_listing
from app.services.extraction import ExtractionError, extract_listing
from evals.metrics import CaseMetrics, aggregate_metrics, evaluate_case

BASE_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"
EXPECTED_FILE = EXAMPLES_DIR / "expected.json"
RESULTS_DIR = BASE_DIR / "evals" / "results"


def load_cases() -> dict:
    with EXPECTED_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def run_case(case_name: str, case: dict) -> tuple[dict, CaseMetrics | None]:
    source_path = EXAMPLES_DIR / case["source_file"]
    raw_text = source_path.read_text(encoding="utf-8")

    try:
        listing = extract_listing(raw_text)
    except ExtractionError as exc:
        print(f"❌ {case_name}: extraction failed: {exc}")

        return {
            "case_name": case_name,
            "passed": False,
            "error": str(exc),
        }, None

    issues = analyze_listing(listing)

    actual_fields = listing.model_dump()
    actual_issue_codes = [issue.code for issue in issues]

    expected_fields = case["expected_fields"]
    expected_issue_codes = case["expected_issue_codes"]

    metrics = evaluate_case(
        expected_fields=expected_fields,
        actual_fields=actual_fields,
        expected_issue_codes=expected_issue_codes,
        actual_issue_codes=actual_issue_codes,
    )

    field_mismatches = {
        field: {
            "expected": expected_value,
            "actual": actual_fields.get(field),
        }
        for field, expected_value in expected_fields.items()
        if actual_fields.get(field) != expected_value
    }

    missed_issues = sorted(set(expected_issue_codes) - set(actual_issue_codes))
    extra_issues = sorted(set(actual_issue_codes) - set(expected_issue_codes))

    passed = not field_mismatches and not missed_issues and not extra_issues

    print(f"{'✅' if passed else '❌'} {case_name}")

    for field, values in field_mismatches.items():
        print(f"   {field}: {values}")

    if missed_issues:
        print(f"   Missed issues: {missed_issues}")

    if extra_issues:
        print(f"   Extra issues: {extra_issues}")

    result = {
        "case_name": case_name,
        "passed": passed,
        "expected_fields": expected_fields,
        "actual_fields": actual_fields,
        "expected_issue_codes": expected_issue_codes,
        "actual_issue_codes": actual_issue_codes,
        "field_mismatches": field_mismatches,
        "missed_issues": missed_issues,
        "extra_issues": extra_issues,
        "metrics": {
            "matched_fields": metrics.matched_fields,
            "total_fields": metrics.total_fields,
            "tp": metrics.true_positives,
            "fp": metrics.false_positives,
            "fn": metrics.false_negatives,
        },
    }

    return result, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Re-evaluate saved predictions without calling Gemini",
    )
    parser.add_argument(
        "--case",
        nargs="+",
        help="Evaluate only selected case names",
    )

    args = parser.parse_args()

    cases = load_cases()

    if args.case:
        unknown_cases = set(args.case) - set(cases)

        if unknown_cases:
            parser.error(f"Unknown cases: {', '.join(sorted(unknown_cases))}")

        cases = {name: case for name, case in cases.items() if name in args.case}

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RESULTS_DIR / "baseline.json"

    if args.replay:
        if not output_path.exists():
            parser.error("baseline.json does not exist. Run a live evaluation first.")

        with output_path.open(encoding="utf-8") as file:
            saved = json.load(file)

        saved_cases = {
            result["case_name"]: result
            for result in saved["cases"]
            if "actual_fields" in result
        }

        if not saved_cases:
            parser.error("No saved predictions found in baseline.json.")

        results = []
        case_metrics = []

        for case_name, case in cases.items():
            if case_name not in saved_cases:
                print(f"SKIP {case_name}: no saved prediction")
                continue

            actual_fields = saved_cases[case_name]["actual_fields"]

            listing = Listing.model_validate(actual_fields)
            issues = analyze_listing(listing)
            actual_issue_codes = [issue.code for issue in issues]

            metrics = evaluate_case(
                expected_fields=case["expected_fields"],
                actual_fields=actual_fields,
                expected_issue_codes=case["expected_issue_codes"],
                actual_issue_codes=actual_issue_codes,
            )

            mismatches = {
                field: {
                    "expected": value,
                    "actual": actual_fields.get(field),
                }
                for field, value in case["expected_fields"].items()
                if actual_fields.get(field) != value
            }

            missed = sorted(set(case["expected_issue_codes"]) - set(actual_issue_codes))
            extra = sorted(set(actual_issue_codes) - set(case["expected_issue_codes"]))

            passed = not mismatches and not missed and not extra

            results.append(
                {
                    "case_name": case_name,
                    "passed": passed,
                    "expected_fields": case["expected_fields"],
                    "actual_fields": actual_fields,
                    "expected_issue_codes": case["expected_issue_codes"],
                    "actual_issue_codes": actual_issue_codes,
                    "field_mismatches": mismatches,
                    "missed_issues": missed,
                    "extra_issues": extra,
                    "metrics": {
                        "matched_fields": metrics.matched_fields,
                        "total_fields": metrics.total_fields,
                        "tp": metrics.true_positives,
                        "fp": metrics.false_positives,
                        "fn": metrics.false_negatives,
                    },
                }
            )
            case_metrics.append(metrics)

        summary = aggregate_metrics(case_metrics)
        passed_count = sum(result["passed"] for result in results)

        print("\n=== Replay Evaluation Summary ===")
        print(f"Cases passed:   {passed_count}/{len(cases)}")
        print(f"Evaluated:      {len(results)}/{len(cases)}")
        print(f"Field accuracy: {summary['field_accuracy']:.2%}")
        print(f"Precision:      {summary['precision']:.2%}")
        print(f"Recall:         {summary['recall']:.2%}")
        print(f"F1:             {summary['f1']:.2%}")

        replay_path = RESULTS_DIR / "replay.json"
        with replay_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "created_at": datetime.now(UTC).isoformat(),
                    "total_cases": len(cases),
                    "evaluated_cases": len(results),
                    "passed_cases": passed_count,
                    "summary": summary,
                    "cases": results,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nSaved: {replay_path}")
        return

    # 기존 Live Evaluation
    results = []
    case_metrics = []

    for case_name, case in cases.items():
        result, metrics = run_case(case_name, case)
        results.append(result)

        if metrics is not None:
            case_metrics.append(metrics)

    summary = aggregate_metrics(case_metrics)
    passed_count = sum(result["passed"] for result in results)

    print("\n=== Live Evaluation Summary ===")
    print(f"Cases passed:   {passed_count}/{len(cases)}")
    print(f"Evaluated:      {len(case_metrics)}/{len(cases)}")
    print(f"Field accuracy: {summary['field_accuracy']:.2%}")
    print(f"Precision:      {summary['precision']:.2%}")
    print(f"Recall:         {summary['recall']:.2%}")
    print(f"F1:             {summary['f1']:.2%}")

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "created_at": datetime.now(UTC).isoformat(),
                "total_cases": len(cases),
                "evaluated_cases": len(case_metrics),
                "passed_cases": passed_count,
                "summary": summary,
                "cases": results,
            },
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
