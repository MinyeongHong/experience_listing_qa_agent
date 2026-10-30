from evals.models import SemanticEvalError, SemanticEvalResult, SemanticMetrics


def calculate_metrics(results: list[SemanticEvalResult]) -> SemanticMetrics:
    tp = fp = fn = tn = 0

    for result in results:
        if isinstance(result, SemanticEvalError):
            continue

        expected = result.expected_review
        predicted = result.predicted_review

        if expected and predicted:
            tp += 1

        elif not expected and predicted:
            fp += 1

        elif expected and not predicted:
            fn += 1

        else:
            tn += 1

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp + fn == 0:
        recall = 0
    else:
        recall = tp / (tp + fn)

    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return SemanticMetrics(
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
        precision=precision,
        recall=recall,
        f1=f1,
    )
