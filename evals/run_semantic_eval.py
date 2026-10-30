import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.domain.models import SemanticDecision
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.services.review import SemanticClassificationError
from evals.error_analysis import print_error_analysis
from evals.models import (
    EvalRunMetadata,
    SemanticEvalCase,
    SemanticEvalError,
    SemanticEvalResult,
    SemanticEvalSuccess,
    SemanticMetrics,
    parse_eval_cases,
    parse_eval_results,
)
from evals.report import build_eval_report
from evals.semantic_metrics import calculate_metrics


def count_errors(results: list[SemanticEvalResult]) -> int:
    count = 0

    for result in results:
        if isinstance(result, SemanticEvalError):
            count += 1

    return count


def print_metrics(
    mode: str,
    metrics: SemanticMetrics,
    evaluated: int,
) -> None:

    print(f"\n=== Semantic Review Evaluation ({mode}) ===")
    print(f"TP={metrics.tp}, FP={metrics.fp}")
    print(f"FN={metrics.fn}, TN={metrics.tn}")
    print(f"Precision={metrics.precision:.2%}")
    print(f"Recall={metrics.recall:.2%}")
    print(f"F1={metrics.f1:.2%}")
    print(f"Evaluated={evaluated}")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--replay", action="store_true")
    parser.add_argument(
        "--results-path",
        type=Path,
        default=Path("evals/semantic_results.json"),
    )

    args = parser.parse_args()

    if args.replay:
        results_path = args.results_path

        raw_results: list[dict] = json.loads(results_path.read_text(encoding="utf-8"))

        evaluation_results: list[SemanticEvalResult] = parse_eval_results(raw_results)

        print_error_analysis(evaluation_results)

        report = build_eval_report(evaluation_results)

        metrics = report.metrics

        evaluated_sum: int = sum(
            1
            for result in evaluation_results
            if isinstance(result, SemanticEvalSuccess)
        )

        print_metrics(
            mode="Replay",
            metrics=metrics,
            evaluated=evaluated_sum,
        )

        print(f"Errors={count_errors(evaluation_results)}")

        return

    load_dotenv()

    raw_cases = json.loads(
        Path("evals/semantic_cases.json").read_text(encoding="utf-8")
    )

    cases: list[SemanticEvalCase] = parse_eval_cases(raw_cases)

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key or not model:
        raise RuntimeError("Gemini configuration is missing.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(timeout=30_000),
    )

    classifier = GeminiSemanticClassifier(
        client=client,
        model=model,
    )

    try:
        skipped = 0
        evaluation_results: list[SemanticEvalResult] = []

        for case in cases:
            if case.included.strip().casefold() == case.excluded.strip().casefold():
                skipped += 1
                print(f"{case.id}: SKIPPED (deterministic rule)")
                continue

            try:
                decision: SemanticDecision = classifier.classify(
                    included_item=case.included,
                    excluded_item=case.excluded,
                )

                predicted_review = decision.verdict != "no_conflict"

            except SemanticClassificationError as exc:
                evaluation_results.append(
                    SemanticEvalError(
                        id=case.id,
                        expected_review=case.expected_review,
                        error=str(exc),
                    )
                )

                print(f"{case.id}: ERROR ({exc})")

                continue

            expected_review = case.expected_review

            evaluation_results.append(
                SemanticEvalSuccess(
                    id=case.id,
                    expected_review=case.expected_review,
                    predicted_review=predicted_review,
                    verdict=decision.verdict,
                    reason=decision.reason,
                )
            )

            print(
                f"{case.id}: "
                f"expected={expected_review}, "
                f"predicted={predicted_review}, "
                f"verdict={decision.verdict}"
            )

        print_error_analysis(evaluation_results)

        metrics = calculate_metrics(evaluation_results)

        evaluated_sum = sum(
            1
            for result in evaluation_results
            if isinstance(result, SemanticEvalSuccess)
        )

        print_metrics(
            mode="Live",
            metrics=metrics,
            evaluated=evaluated_sum,
        )

        print(f"Errors={count_errors(evaluation_results)}")
        print(f"Skipped={skipped}")

        output_path = Path("evals/semantic_results.json")

        output_path.write_text(
            json.dumps(
                [result.model_dump() for result in evaluation_results],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Results saved to {output_path}")

        report = build_eval_report(
            evaluation_results,
            total_cases=len(cases),
            skipped_cases=skipped,
            metadata=EvalRunMetadata(
                model=model,
                prompt_version="v1",
                dataset_version="v1",
                mode="live",
                timestamp=datetime.now(UTC).isoformat(),
            ),
        )

        report_path = Path("evals/semantic_report.json")

        report_path.write_text(
            json.dumps(
                report.model_dump_json(indent=2),
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"Reports saved to {report_path}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
