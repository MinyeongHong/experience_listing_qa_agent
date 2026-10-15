from dataclasses import dataclass


@dataclass
class CaseMetrics:
    matched_fields: int
    total_fields: int
    true_positives: int
    false_positives: int
    false_negatives: int

    @property
    def field_accuracy(self) -> float:
        if self.total_fields == 0:
            return 1.0
        return self.matched_fields / self.total_fields

    @property
    def precision(self) -> float:
        denominator = self.true_positives + self.false_positives
        if denominator == 0:
            return 1.0
        return self.true_positives / denominator

    @property
    def recall(self) -> float:
        denominator = self.true_positives + self.false_negatives
        if denominator == 0:
            return 1.0
        return self.true_positives / denominator

    @property
    def f1(self) -> float:
        precision = self.precision
        recall = self.recall

        if precision + recall == 0:
            return 0.0

        return 2 * precision * recall / (precision + recall)


def evaluate_case(
    expected_fields: dict,
    actual_fields: dict,
    expected_issue_codes: list[str],
    actual_issue_codes: list[str],
) -> CaseMetrics:
    matched_fields = sum(
        actual_fields.get(field) == expected_value
        for field, expected_value in expected_fields.items()
    )

    expected_issues = set(expected_issue_codes)
    actual_issues = set(actual_issue_codes)

    return CaseMetrics(
        matched_fields=matched_fields,
        total_fields=len(expected_fields),
        true_positives=len(expected_issues & actual_issues),
        false_positives=len(actual_issues - expected_issues),
        false_negatives=len(expected_issues - actual_issues),
    )


def aggregate_metrics(results: list[CaseMetrics]) -> dict:
    matched_fields = sum(r.matched_fields for r in results)
    total_fields = sum(r.total_fields for r in results)

    tp = sum(r.true_positives for r in results)
    fp = sum(r.false_positives for r in results)
    fn = sum(r.false_negatives for r in results)

    field_accuracy = matched_fields / total_fields if total_fields else 1.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "matched_fields": matched_fields,
        "total_fields": total_fields,
        "field_accuracy": field_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
