"""Finite-difference check and exact enumeration for a three-action policy.

Python standard library only. This is a mathematical teaching example, not an
LLM training experiment. Run from any directory with Python 3.
"""

import json
import math


def softmax(logits):
    weights = [math.exp(z - max(logits)) for z in logits]
    return [w / sum(weights) for w in weights]


def objective(logits, rewards):
    return sum(p * r for p, r in zip(softmax(logits), rewards))


def sample_gradient(probs, rewards, action, baseline):
    advantage = rewards[action] - baseline
    return [advantage * (int(j == action) - p) for j, p in enumerate(probs)]


def main():
    probs = [0.2, 0.3, 0.5]
    rewards = [1.0, 0.2, 0.0]
    logits = [math.log(p) for p in probs]
    value = objective(logits, rewards)
    analytic = [p * (r - value) for p, r in zip(probs, rewards)]
    eps, lr = 1e-5, 0.5
    finite_difference = []
    for j in range(3):
        plus, minus = logits.copy(), logits.copy()
        plus[j] += eps
        minus[j] -= eps
        finite_difference.append(
            (objective(plus, rewards) - objective(minus, rewards)) / (2 * eps)
        )
    errors = [abs(a - b) for a, b in zip(analytic, finite_difference)]
    assert max(errors) < 1e-8, errors

    baseline_checks = []
    for baseline in (0.0, value, 0.7):
        gradients = [sample_gradient(probs, rewards, i, baseline) for i in range(3)]
        mean = [sum(probs[i] * gradients[i][j] for i in range(3)) for j in range(3)]
        assert max(abs(a - b) for a, b in zip(mean, analytic)) < 1e-12
        variance = sum(
            probs[i] * sum((gradients[i][j] - mean[j]) ** 2 for j in range(3))
            for i in range(3)
        )
        baseline_checks.append({"baseline": baseline, "mean_gradient": mean,
                                "gradient_total_variance": variance})

    def updated(g):
        new_logits = [z + lr * dz for z, dz in zip(logits, g)]
        return {"probabilities": softmax(new_logits),
                "expected_reward": objective(new_logits, rewards)}

    # A single unbiased estimator sample need not improve the true objective.
    sample_b = sample_gradient(probs, rewards, 1, value)
    assert updated(analytic)["expected_reward"] > value
    assert updated(sample_b)["expected_reward"] < value
    print(json.dumps({
        "kind": "analytic_toy_example_not_llm_training",
        "actions": ["A", "B", "C"], "rewards": rewards,
        "initial_probabilities": probs, "initial_expected_reward": value,
        "learning_rate": lr, "exact_gradient": analytic,
        "finite_difference_gradient": finite_difference,
        "max_gradient_error": max(errors),
        "exact_gradient_step": updated(analytic),
        "single_sample_B_with_value_baseline": {
            "gradient": sample_b, **updated(sample_b)},
        "baseline_checks": baseline_checks
    }, indent=2))


if __name__ == "__main__":
    main()
