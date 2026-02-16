"""Statistical power analysis for the A/B experiment.

Computes required sample sizes for comparing treatment (AI-assisted)
vs. control (manual-only) lawyer prediction accuracy.

Key insight from the plan review: the baseline accuracy (55% in the
original framework) is UNSUBSTANTIATED and must be empirically measured
during shadow mode (Phase 5). This module recalculates power with
observed effect sizes.

Usage:
    from src.analysis.power_analysis import compute_sample_size, power_report
    n = compute_sample_size(p_control=0.55, p_treatment=0.65)
    report = power_report(p_control=0.55, p_treatment=0.65)
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class PowerResult:
    """Result of a power analysis computation."""

    n_per_arm: int
    total_n: int
    p_control: float
    p_treatment: float
    absolute_lift_pp: float  # percentage points
    relative_lift_pct: float  # percent
    alpha: float
    power: float
    test_type: str


def compute_sample_size(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> PowerResult:
    """Compute required sample size per arm for a two-proportion Z-test.

    Uses the formula for comparing two independent proportions.

    Args:
        p_control: Expected accuracy in the control arm (manual only)
        p_treatment: Expected accuracy in the treatment arm (AI-assisted)
        alpha: Significance level (Type I error rate)
        power: Statistical power (1 - Type II error rate)
        ratio: Allocation ratio (n_treatment / n_control)

    Returns:
        PowerResult with required sample sizes.
    """
    if p_treatment <= p_control:
        raise ValueError("Treatment proportion must exceed control proportion.")

    # Effect size (Cohen's h)
    h = 2 * np.arcsin(np.sqrt(p_treatment)) - 2 * np.arcsin(np.sqrt(p_control))

    # Z-values
    z_alpha = stats.norm.ppf(1 - alpha / 2)  # two-sided
    z_beta = stats.norm.ppf(power)

    # Sample size per arm (Fleiss formula for two proportions)
    p_bar = (p_control + ratio * p_treatment) / (1 + ratio)
    q_bar = 1 - p_bar

    numerator = (z_alpha * np.sqrt((1 + 1 / ratio) * p_bar * q_bar) +
                 z_beta * np.sqrt(p_control * (1 - p_control) +
                                  p_treatment * (1 - p_treatment) / ratio)) ** 2
    denominator = (p_treatment - p_control) ** 2

    n_control = int(np.ceil(numerator / denominator))
    n_treatment = int(np.ceil(n_control * ratio))

    absolute_lift = (p_treatment - p_control) * 100
    relative_lift = ((p_treatment - p_control) / p_control) * 100

    result = PowerResult(
        n_per_arm=max(n_control, n_treatment),
        total_n=n_control + n_treatment,
        p_control=p_control,
        p_treatment=p_treatment,
        absolute_lift_pp=absolute_lift,
        relative_lift_pct=relative_lift,
        alpha=alpha,
        power=power,
        test_type="two_proportion_z_test",
    )

    logger.info(
        "Power analysis: %.1f%% -> %.1f%% (%.1f pp lift), n=%d per arm, total=%d",
        p_control * 100, p_treatment * 100, absolute_lift, result.n_per_arm, result.total_n,
    )
    return result


def compute_power_curve(
    p_control: float,
    effect_sizes_pp: list[float] | None = None,
    alpha: float = 0.05,
    power: float = 0.80,
) -> list[dict]:
    """Compute sample sizes for a range of effect sizes.

    Useful for planning: "if the true lift is X pp, how many cases do we need?"

    Returns:
        List of dicts with effect_size_pp, n_per_arm, total_n.
    """
    if effect_sizes_pp is None:
        effect_sizes_pp = [3, 5, 7, 10, 12, 15, 20]

    results = []
    for delta in effect_sizes_pp:
        p_treatment = p_control + delta / 100
        if p_treatment >= 1.0:
            continue
        try:
            result = compute_sample_size(p_control, p_treatment, alpha, power)
            results.append({
                "effect_size_pp": delta,
                "n_per_arm": result.n_per_arm,
                "total_n": result.total_n,
            })
        except ValueError:
            continue

    return results


def power_report(
    p_control: float,
    p_treatment: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> str:
    """Generate a human-readable power analysis report.

    Returns:
        Formatted report string.
    """
    result = compute_sample_size(p_control, p_treatment, alpha, power)
    curve = compute_power_curve(p_control, alpha=alpha, power=power)

    lines = [
        "=" * 60,
        "POWER ANALYSIS REPORT",
        "=" * 60,
        f"",
        f"Hypothesis: Treatment accuracy ({p_treatment:.1%}) > Control accuracy ({p_control:.1%})",
        f"Absolute lift: {result.absolute_lift_pp:.1f} percentage points",
        f"Relative lift: {result.relative_lift_pct:.1f}%",
        f"",
        f"Parameters:",
        f"  Significance level (alpha): {alpha}",
        f"  Statistical power (1-beta): {power}",
        f"  Test: Two-proportion Z-test (two-sided)",
        f"",
        f"Required sample size:",
        f"  Per arm: {result.n_per_arm} cases",
        f"  Total (both arms): {result.total_n} cases",
        f"  With 10% dropout buffer: {int(result.total_n * 1.1)} cases",
        f"",
        f"Sensitivity table (baseline = {p_control:.1%}):",
        f"  {'Lift (pp)':>10} | {'N per arm':>10} | {'Total N':>10}",
        f"  {'-' * 10}-+-{'-' * 10}-+-{'-' * 10}",
    ]

    for row in curve:
        lines.append(
            f"  {row['effect_size_pp']:>10.1f} | {row['n_per_arm']:>10d} | {row['total_n']:>10d}"
        )

    lines.extend([
        f"",
        f"IMPORTANT NOTES:",
        f"  - The baseline accuracy ({p_control:.1%}) must be empirically validated",
        f"    during shadow mode (Phase 5), NOT assumed.",
        f"  - These are PROSPECTIVE cases requiring real lawyer predictions,",
        f"    separate from the historical training data.",
        f"  - At ~5-10 new cases per week, collecting {result.n_per_arm} cases per arm",
        f"    takes {result.n_per_arm // 5}-{result.n_per_arm // 10} weeks.",
        "=" * 60,
    ])

    return "\n".join(lines)
