import math

def softmax(scores: list[float]) -> list[float]:
    max_score = max(scores)
    exp_scores = [math.exp(score - max_score) for score in scores]
    sum_exp = sum(exp_scores)

    return [value / sum_exp for value in exp_scores]