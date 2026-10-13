import json
from pathlib import Path

from app.services.analysis import analyze_listing
from app.services.extraction import ExtractionError, extract_listing

BASE_DIR = Path(__file__).resolve().parent.parent
EXAMPLES_DIR = BASE_DIR / "examples"
EXPECTED_FILE = EXAMPLES_DIR / "expected.json"


def load_cases() -> dict:
    with EXPECTED_FILE.open(encoding="utf-8") as file:
        return json.load(file)


def run_case(case_name: str, case: dict) -> bool:
    source_path = EXAMPLES_DIR / case["source_file"]
    raw_text = source_path.read_text(encoding="utf-8")

    try:
        listing = extract_listing(raw_text)
    except ExtractionError as exc:
        print(f"❌ {case_name}: extraction failed: {exc}")
        if exc.__cause__ is not None:
            print(f"   cause: {type(exc.__cause__).__name__}: {exc.__cause__}")

            return False

    issues = analyze_listing(listing)

    actual_fields = listing.model_dump()
    actual_issue_codes = [issue.code for issue in issues]

    passed = True

    for field, expected_value in case["expected_fields"].items():
        actual_value = actual_fields[field]

        if actual_value != expected_value:
            passed = False
            print(f"  ❌ {field}")
            print(f"     expected: {expected_value}")
            print(f"     actual:   {actual_value}")

    expected_issue_codes = case["expected_issue_codes"]

    if set(actual_issue_codes) != set(expected_issue_codes):
        passed = False
        print("  ❌ issue codes")
        print(f"     expected: {expected_issue_codes}")
        print(f"     actual:   {actual_issue_codes}")

    if passed:
        print(f"✅ {case_name}")

    return passed


def main():
    cases = load_cases()

    passed_count = 0

    for case_name, case in cases.items():
        if run_case(case_name, case):
            passed_count += 1

    total = len(cases)

    print()
    print(f"Result: {passed_count}/{total} cases passed")


if __name__ == "__main__":
    main()
