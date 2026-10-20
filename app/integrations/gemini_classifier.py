from typing import Any

from google.genai import errors, types
from pydantic import ValidationError

from app.domain.models import SemanticDecision
from app.services.review import SemanticClassificationError

SYSTEM_INSTRUCTION = """
You review potential contradictions in travel experience listings.

Compare an included item with an excluded item.

Return:
- possible_conflict: likely the same or overlapping service
- no_conflict: no meaningful contradiction is supported
- uncertain: insufficient information to decide

Consider synonyms, plurals, and differences in service scope.
Do not assume that related concepts are always identical.
When uncertain, choose uncertain.

Provide a concise reason.
"""


class GeminiSemanticClassifier:
    def __init__(
        self,
        client: Any,
        model: str = "gemini-2.5-flash",
    ):
        self.client = client
        self.model = model

    def classify(
        self,
        included_item: str,
        excluded_item: str,
    ) -> SemanticDecision:
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=(
                    f"Included item: {included_item}\nExcluded item: {excluded_item}"
                ),
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    response_mime_type="application/json",
                    response_schema=SemanticDecision,
                    temperature=0,
                ),
            )

            parsed = response.parsed

            if isinstance(parsed, SemanticDecision):
                return parsed

            if parsed is None:
                raise SemanticClassificationError(
                    "Gemini returned no structured decision."
                )

            return SemanticDecision.model_validate(parsed)

        except errors.APIError as exc:
            raise SemanticClassificationError("Gemini API request failed.") from exc

        except (ValidationError, ValueError, TypeError) as exc:
            raise SemanticClassificationError(
                "Gemini returned an invalid decision."
            ) from exc
