import os

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types

from app.domain.models import Listing

load_dotenv()

client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options=types.HttpOptions(
        timeout=30_000,
    ),
)
model = os.getenv("GEMINI_MODEL")


class ExtractionError(Exception):
    pass


class RateLimitError(ExtractionError):
    pass


class ExtractionTimeoutError(ExtractionError):
    pass


def extract_listing(raw_text: str) -> Listing:
    try:
        response = client.models.generate_content(
            model=model,
            contents=f"""
Extract the experience listing information from the text below.

Extract only information explicitly present in the source text.
Do not correct, reconcile, deduplicate, or resolve contradictions.

If the same item appears in both inclusions and exclusions,
preserve it in both fields exactly as stated.

If information is missing, do not invent it.

For tags, only extract tags if the source text contains an explicit
tag or category section. Do not infer or generate tags from the description.
If no explicit tags are present, return an empty list.

Only extract meeting_point when the source explicitly identifies a
specific meeting location or named place where guests should meet.
Do not treat vague route descriptions such as "near a major city square",
"in the city centre", or "in the heart of the city" as a meeting point.
If no specific meeting point is provided, return null.

Preserve all explicitly stated duration values in duration_mentions,
converted to minutes.

Listing:
{raw_text}
""",
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=Listing,
            ),
        )

    except (TimeoutError, httpx.TimeoutException) as exc:
        raise ExtractionTimeoutError("Gemini request timed out.") from exc

    except errors.ClientError as exc:
        if exc.code == 429:
            raise RateLimitError("Gemini rate limit exceeded.") from exc

        raise ExtractionError("Failed to extract listing.") from exc

    except Exception as exc:
        raise ExtractionError("Failed to extract listing.") from exc

    if response.parsed is None:
        raise ExtractionError("Gemini returned an invalid structured response.")

    return response.parsed
