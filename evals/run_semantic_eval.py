import json
import os
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.services.review import SemanticClassificationError


def main():
    load_dotenv()

    cases = json.loads(Path("evals/semantic_cases.json").read_text(encoding="utf-8"))

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

    tp = fp = fn = tn = 0
    errors = 0
    skipped = 0

    try:
        for case in cases:
            # Exact matches are handled by deterministic rules.
            if (
                case["included"].strip().casefold()
                == case["excluded"].strip().casefold()
            ):
                skipped += 1
                print(f"{case['id']}: SKIPPED (deterministic rule)")
                continue

            try:
                decision = classifier.classify(
                    included_item=case["included"],
                    excluded_item=case["excluded"],
                )

                predicted_review = decision.verdict != "no_conflict"

            except SemanticClassificationError as exc:
                errors += 1
                print(f"{case['id']}: ERROR ({exc})")
                continue

            expected_review = case["expected_review"]

            if predicted_review and expected_review:
                tp += 1
            elif predicted_review and not expected_review:
                fp += 1
            elif not predicted_review and expected_review:
                fn += 1
            else:
                tn += 1

            print(
                f"{case['id']}: "
                f"expected={expected_review}, "
                f"predicted={predicted_review}, "
                f"verdict={decision.verdict}"
            )

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

        print("\n=== Semantic Review Evaluation ===")
        print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
        print(f"Errors={errors}")
        print(f"Precision={precision:.2%}")
        print(f"Recall={recall:.2%}")
        print(f"F1={f1:.2%}")
        print(f"Evaluated={tp + fp + fn + tn}")
        print(f"Skipped={skipped}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
