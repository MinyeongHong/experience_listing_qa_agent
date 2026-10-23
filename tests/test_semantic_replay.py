import json
import sys

from evals import run_semantic_eval


def fail_if_called(*args, **kwargs):
    raise AssertionError("Gemini client must not be created during replay")


def test_replay_does_not_create_gemini_client(
    monkeypatch,
    tmp_path,
    capsys,
):
    results = [
        {
            "id": "case_1",
            "expected_review": True,
            "predicted_review": True,
            "verdict": "possible_conflict",
            "reason": "Overlapping services",
        }
    ]

    results_path = tmp_path / "semantic_results.json"
    results_path.write_text(
        json.dumps(results),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_semantic_eval",
            "--replay",
            "--results-path",
            str(results_path),
        ],
    )

    monkeypatch.setattr(
        run_semantic_eval.genai,
        "Client",
        fail_if_called,
    )

    # 1. 테스트할 함수 실행
    run_semantic_eval.main()

    # 2. print()로 출력된 내용 가져오기
    output = capsys.readouterr().out

    # 3. 출력 결과 검증
    assert "TP=1" in output


def test_replay_excludes_errors_from_metrics(
    monkeypatch,
    tmp_path,
    capsys,
):
    results = [
        {
            "id": "success_case",
            "status": "success",
            "expected_review": True,
            "predicted_review": True,
            "verdict": "possible_conflict",
            "reason": "Overlapping services",
        },
        {
            "id": "error_case",
            "status": "error",
            "expected_review": True,
            "error": "Gemini API failed",
        },
    ]

    results_path = tmp_path / "semantic_results.json"
    results_path.write_text(
        json.dumps(results),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_semantic_eval",
            "--replay",
            "--results-path",
            str(results_path),
        ],
    )

    monkeypatch.setattr(
        run_semantic_eval.genai,
        "Client",
        fail_if_called,
    )

    # 1. 테스트할 함수 실행
    run_semantic_eval.main()

    # 2. print()로 출력된 내용 가져오기
    output = capsys.readouterr().out

    # 3. 출력 결과 검증
    assert "TP=1" in output
    assert "Evaluated=1" in output
    assert "Errors=1" in output
    assert "FN=0, TN=0" in output
