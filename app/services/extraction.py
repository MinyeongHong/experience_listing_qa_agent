import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.domain.models import Listing

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
model = os.getenv("GEMINI_MODEL")


def extract_listing(raw_text: str) -> Listing:
    response = client.models.generate_content(
        model=model,
        contents=f"""
Extract the experience listing information from the following text.

Listing:
{raw_text}
""",
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=Listing,
        ),
    )

    return response.parsed
