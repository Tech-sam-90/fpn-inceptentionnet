"""
Statistical testing utilities to address reviewer W7.

Functions
---------
bootstrap_ci       : 95% CI on a metric across folds via bootstrap resampling
wilcoxon_test      : Wilcoxon signed-rank test comparing per-fold metrics
mcnemar_test       : McNemar's test comparing per-sample predictions
compare_models     : Full table comparing two sets of fold results
"""
from __future__ import annotations

import numpy as np
from scipy import stats


def bootstrap_ci(
    values: np.ndarray,
    n_bootstrap: int = 10_000,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    """
    Bootstrap confidence interval for the mean of `values`.
    Returns (lower, upper) bounds.
    """
    rng = np.random.default_rng(seed)
    means = np.array([rng.choice(values, size=len(values), replace=True).mean()
                      for _ in range(n_bootstrap)])
    alpha = (1 - ci) / 2
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1 - alpha))


def wilcoxon_test(
    a: np.ndarray,
    b: np.ndarray,
) -> tuple[float, float]:
    """
    Wilcoxon signed-rank test between paired per-fold metric arrays.
    Returns (statistic, p_value). p < 0.05 → reject null of equal medians.
    """
    if len(a) < 3:
        return float("nan"), float("nan")
    result = stats.wilcoxon(a, b, alternative="two-sided", zero_method="zsplit")
    return float(result.statistic), float(result.pvalue)


def mcnemar_test(
    labels: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
) -> tuple[float, float]:
    """
    McNemar's test on paired binary predictions from two models.
    Returns (chi2, p_value).
    """
    a_right = preds_a == labels
    b_right = preds_b == labels
    # Discordant cells
    n_01 = int(((~a_right) & b_right).sum())   # A wrong, B right
    n_10 = int((a_right & (~b_right)).sum())    # A right, B wrong
    total = n_01 + n_10
    if total == 0:
        return 0.0, 1.0
    # Exact McNemar (binomial)
    p_value = float(2 * min(
        stats.binom.cdf(min(n_01, n_10), total, 0.5),
        1 - stats.binom.cdf(min(n_01, n_10) - 1, total, 0.5),
    ))
    chi2 = float((abs(n_01 - n_10) - 1) ** 2 / total) if total > 0 else 0.0
    return chi2, p_value


def compare_models(
    fold_results_a: list[dict],
    fold_results_b: list[dict],
    name_a: str = "Model A",
    name_b: str = "Model B",
    metrics: list[str] | None = None,
) -> dict:
    """
    Returns a comparison table with per-metric means, CIs, and Wilcoxon p-values.
    """
    if metrics is None:
        metrics = ["accuracy", "precision", "recall", "sensitivity", "specificity", "f1", "auc"]

    table = {}
    for m in metrics:
        vals_a = np.array([fr["metrics"].get(m, float("nan")) for fr in fold_results_a])
        vals_b = np.array([fr["metrics"].get(m, float("nan")) for fr in fold_results_b])
        vals_a = vals_a[np.isfinite(vals_a)]
        vals_b = vals_b[np.isfinite(vals_b)]

        ci_a = bootstrap_ci(vals_a) if len(vals_a) > 0 else (float("nan"), float("nan"))
        ci_b = bootstrap_ci(vals_b) if len(vals_b) > 0 else (float("nan"), float("nan"))
        stat, pval = wilcoxon_test(vals_a, vals_b) if len(vals_a) >= 3 and len(vals_a) == len(vals_b) else (float("nan"), float("nan"))

        table[m] = {
            f"{name_a}_mean": float(vals_a.mean()) if len(vals_a) > 0 else float("nan"),
            f"{name_a}_ci95": ci_a,
            f"{name_b}_mean": float(vals_b.mean()) if len(vals_b) > 0 else float("nan"),
            f"{name_b}_ci95": ci_b,
            "delta": float(vals_b.mean() - vals_a.mean()) if len(vals_a) > 0 and len(vals_b) > 0 else float("nan"),
            "wilcoxon_stat": stat,
            "wilcoxon_p": pval,
            "significant": pval < 0.05 if np.isfinite(pval) else False,
        }
    return table


def print_comparison_table(table: dict, name_a: str = "Baseline", name_b: str = "Ours") -> None:
    header = f"{'Metric':<14} {name_a:>10} (95% CI)           {name_b:>10} (95% CI)           {'Delta':>8}  {'p-value':>8}  Sig?"
    print(header)
    print("-" * len(header))
    for m, row in table.items():
        a_ci = row[f"{name_a}_ci95"] if f"{name_a}_ci95" in row else (float("nan"), float("nan"))
        b_ci = row[f"{name_b}_ci95"] if f"{name_b}_ci95" in row else (float("nan"), float("nan"))
        sig = "*" if row.get("significant") else ""
        print(
            f"{m:<14} "
            f"{row.get(f'{name_a}_mean', float('nan')):>8.4f} [{a_ci[0]:.4f}–{a_ci[1]:.4f}]  "
            f"{row.get(f'{name_b}_mean', float('nan')):>8.4f} [{b_ci[0]:.4f}–{b_ci[1]:.4f}]  "
            f"{row.get('delta', float('nan')):>+8.4f}  "
            f"{row.get('wilcoxon_p', float('nan')):>8.4f}  {sig}"
        )
