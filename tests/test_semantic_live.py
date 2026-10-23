import json
import sys
from unittest.mock import MagicMock

from app.services.review import SemanticClassificationError
from evals import run_semantic_eval


class FakeClassifier:
    def __init__(self, client, model):
        pass

    def classify(self, included_item, excluded_item):
        raise SemanticClassificationError("Simulated Gemini failure")


def test_live_records_classification_errors(
    monkeypatch,
    tmp_path,
    capsys,
):
    fake_client = MagicMock()

    # 1. 실제 Gemini 객체를 가짜 객체로 교체
    monkeypatch.setattr(
        run_semantic_eval.genai,
        "Client",
        lambda **kwargs: fake_client,
    )

    monkeypatch.setattr(
        run_semantic_eval,
        "GeminiSemanticClassifier",
        FakeClassifier,
    )

    # 2. 환경변수와 명령행 인자 준비
    monkeypatch.setenv("GEMINI_API_KEY", "fake-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "fake-model")
    monkeypatch.setattr(sys, "argv", ["run_semantic_eval"])

    # 3. 임시 작업 디렉터리로 이동
    monkeypatch.chdir(tmp_path)

    # 4. 테스트용 평가 케이스 생성
    cases = [
        {
            "id": "failure_case",
            "included": "Food tasting",
            "excluded": "Food tastings",
            "expected_review": True,
        }
    ]

    evals_dir = tmp_path / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)

    cases_path = evals_dir / "semantic_cases.json"
    cases_path.write_text(
        json.dumps(cases),
        encoding="utf-8",
    )

    # 5. Live Evaluation 실행
    run_semantic_eval.main()

    # 6. 출력 검증
    output = capsys.readouterr().out

    assert "Errors=1" in output
    assert "Evaluated=0" in output

    # 7. 저장된 JSON 검증
    results_path = evals_dir / "semantic_results.json"

    saved_results = json.loads(results_path.read_text(encoding="utf-8"))

    assert len(saved_results) == 1
    assert saved_results[0]["status"] == "error"
    assert saved_results[0]["id"] == "failure_case"
    assert saved_results[0]["error"] == "Simulated Gemini failure"

    # 8. Gemini 클라이언트 정리 확인
    fake_client.close.assert_called_once()
