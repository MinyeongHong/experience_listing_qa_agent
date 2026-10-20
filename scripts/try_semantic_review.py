import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.domain.models import Listing
from app.integrations.gemini_classifier import GeminiSemanticClassifier
from app.services.review import find_review_candidates


def main():
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    model = os.getenv("GEMINI_MODEL")

    if not api_key or not model:
        raise RuntimeError("GEMINI_API_KEY and GEMINI_MODEL must be configured.")

    client = genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            timeout=30_000,
        ),
    )

    try:
        classifier = GeminiSemanticClassifier(
            client=client,
            model=model,
        )

        listing = Listing(
            title="Barcelona Food Walking Tour",
            city="Barcelona",
            duration_minutes=120,
            meeting_point="Plaça de Catalunya",
            inclusions=["Food tasting"],
            exclusions=["Food tastings"],
        )

        decision = classifier.classify(
            included_item="Food tasting",
            excluded_item="Food tastings",
        )

        print("=== Semantic Decision ===")
        print(decision.model_dump_json(indent=2))

        candidates = find_review_candidates(
            listing=listing,
            classifier=classifier,
        )

        print("=== Review Candidates ===")

        for candidate in candidates:
            print(candidate.model_dump_json(indent=2))

        if not candidates:
            print("No review candidates found.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
