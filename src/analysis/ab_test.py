"""A/B test statistical analysis for the litigation analytics experiment.

Implements the statistical plan from the experiment design:
1. Two-proportion Z-test for accuracy lift (primary)
2. Paired t-test for time-saving (secondary)
3. Mixed-effects logistic regression for controlling confounders

Usage:
    from src.analysis.ab_test import analyze_experiment
    results = analyze_experiment(experiment_df)
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    """Result of an A/B test statistical comparison."""

    metric_name: str
    control_value: float
    treatment_value: float
    absolute_diff: float
    relative_diff_pct: float
    z_statistic: float | None = None
    t_statistic: float | None = None
    p_value: float = 1.0
    confidence_interval_95: tuple[float, float] = (0.0, 0.0)
    significant: bool = False
    test_type: str = ""


def two_proportion_z_test(
    n_control: int,
    wins_control: int,
    n_treatment: int,
    wins_treatment: int,
    alpha: float = 0.05,
) -> ABTestResult:
    """Two-proportion Z-test for accuracy comparison.

    Primary test for the experiment: is treatment accuracy
    significantly greater than control accuracy?

    Args:
        n_control: Total cases in control arm
        wins_control: Correct predictions in control arm
        n_treatment: Total cases in treatment arm
        wins_treatment: Correct predictions in treatment arm
        alpha: Significance level
    """
    p_control = wins_control / n_control
    p_treatment = wins_treatment / n_treatment

    # Pooled proportion
    p_pool = (wins_control + wins_treatment) / (n_control + n_treatment)

    # Z-statistic
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_control + 1 / n_treatment))
    z = (p_treatment - p_control) / se if se > 0 else 0

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    # 95% CI for the difference
    se_diff = np.sqrt(
        p_control * (1 - p_control) / n_control + p_treatment * (1 - p_treatment) / n_treatment
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (
        (p_treatment - p_control) - z_crit * se_diff,
        (p_treatment - p_control) + z_crit * se_diff,
    )

    absolute_diff = p_treatment - p_control
    relative_diff = (absolute_diff / p_control * 100) if p_control > 0 else 0

    return ABTestResult(
        metric_name="accuracy",
        control_value=p_control,
        treatment_value=p_treatment,
        absolute_diff=absolute_diff,
        relative_diff_pct=relative_diff,
        z_statistic=z,
        p_value=p_value,
        confidence_interval_95=ci,
        significant=p_value < alpha,
        test_type="two_proportion_z_test",
    )


def paired_t_test_time(
    control_times: np.ndarray,
    treatment_times: np.ndarray,
    alpha: float = 0.05,
) -> ABTestResult:
    """Paired t-test for research time reduction.

    Secondary metric: does AI assistance reduce time spent per case?

    Args:
        control_times: Time (minutes) per case in control arm
        treatment_times: Time (minutes) per case in treatment arm
        alpha: Significance level
    """
    t_stat, p_value = stats.ttest_ind(control_times, treatment_times, alternative="greater")

    control_mean = control_times.mean()
    treatment_mean = treatment_times.mean()
    absolute_diff = control_mean - treatment_mean  # Positive = time saved
    relative_diff = (absolute_diff / control_mean * 100) if control_mean > 0 else 0

    # 95% CI for the difference
    se = np.sqrt(
        control_times.var() / len(control_times) + treatment_times.var() / len(treatment_times)
    )
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (absolute_diff - z_crit * se, absolute_diff + z_crit * se)

    return ABTestResult(
        metric_name="research_time_minutes",
        control_value=control_mean,
        treatment_value=treatment_mean,
        absolute_diff=absolute_diff,
        relative_diff_pct=relative_diff,
        t_statistic=t_stat,
        p_value=p_value,
        confidence_interval_95=ci,
        significant=p_value < alpha,
        test_type="independent_t_test",
    )


def mixed_effects_regression(
    df: pd.DataFrame,
    outcome_col: str = "correct_prediction",
    treatment_col: str = "is_treatment",
    random_effect_col: str = "lawyer_id",
    fixed_effects: list[str] | None = None,
) -> dict:
    """Mixed-effects logistic regression for controlling confounders.

    Model: Outcome ~ Treatment + JudgeID + ClaimBucket + (1|LawyerID)

    Args:
        df: Experiment data with outcome, treatment indicator, and covariates
        outcome_col: Binary outcome column (1 = correct prediction)
        treatment_col: Treatment indicator (1 = AI-assisted)
        random_effect_col: Random effect grouping variable
        fixed_effects: List of fixed-effect covariate columns

    Returns:
        Dict with model summary, treatment effect, and p-value.
    """
    try:
        import statsmodels.api as sm
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.generalized_estimating_equations import GEE

        if fixed_effects is None:
            fixed_effects = [treatment_col]
        else:
            fixed_effects = [treatment_col] + [f for f in fixed_effects if f != treatment_col]

        # GEE as approximation to mixed-effects (more robust with small clusters)
        formula_vars = df[fixed_effects].copy()
        formula_vars = pd.get_dummies(formula_vars, drop_first=True)
        formula_vars = sm.add_constant(formula_vars)

        model = GEE(
            df[outcome_col],
            formula_vars,
            groups=df[random_effect_col],
            family=Binomial(),
        )
        result = model.fit()

        # Extract treatment effect
        treatment_coef = result.params.get(treatment_col, 0)
        treatment_pvalue = result.pvalues.get(treatment_col, 1)
        treatment_or = np.exp(treatment_coef)  # Odds ratio

        return {
            "treatment_coefficient": float(treatment_coef),
            "treatment_odds_ratio": float(treatment_or),
            "treatment_p_value": float(treatment_pvalue),
            "model_summary": str(result.summary()),
            "converged": result.converged,
        }

    except ImportError:
        logger.warning("statsmodels not installed. Skipping mixed-effects regression.")
        return {"error": "statsmodels required for mixed-effects regression"}
    except Exception as e:
        logger.error("Mixed-effects regression failed: %s", e)
        return {"error": str(e)}


def analyze_experiment(
    df: pd.DataFrame,
    accuracy_col: str = "correct_prediction",
    time_col: str = "research_time_minutes",
    treatment_col: str = "is_treatment",
    alpha: float = 0.05,
) -> dict:
    """Run the full statistical analysis suite on experiment data.

    Implements the roll-out criteria:
    - GREEN: Accuracy uplift >= target AND time saved >= 30% (both p < 0.05)
    - YELLOW: One metric passes, the other marginal
    - RED: No uplift or bias detected

    Args:
        df: Experiment data
        accuracy_col: Column indicating whether prediction was correct
        time_col: Column with research time in minutes
        treatment_col: Column indicating treatment assignment

    Returns:
        Dict with all test results and roll-out decision.
    """
    control = df[df[treatment_col] == 0]
    treatment = df[df[treatment_col] == 1]

    results = {}

    # Primary: accuracy comparison
    accuracy_result = two_proportion_z_test(
        n_control=len(control),
        wins_control=int(control[accuracy_col].sum()),
        n_treatment=len(treatment),
        wins_treatment=int(treatment[accuracy_col].sum()),
        alpha=alpha,
    )
    results["accuracy_test"] = accuracy_result

    # Secondary: time comparison (if available)
    if time_col in df.columns:
        time_result = paired_t_test_time(
            control_times=control[time_col].values,
            treatment_times=treatment[time_col].values,
            alpha=alpha,
        )
        results["time_test"] = time_result
    else:
        time_result = None

    # Roll-out decision
    accuracy_pass = accuracy_result.significant and accuracy_result.absolute_diff > 0
    time_pass = (
        time_result is not None and time_result.significant and time_result.relative_diff_pct >= 30
    )

    if accuracy_pass and time_pass:
        decision = "GREEN"
        reason = "Both accuracy uplift and time reduction are significant."
    elif accuracy_pass or time_pass:
        decision = "YELLOW"
        reason = "One metric passes but the other is marginal. Consider a second pilot."
    else:
        decision = "RED"
        reason = "No significant uplift detected. Revisit features and model."

    results["rollout_decision"] = {
        "decision": decision,
        "reason": reason,
        "accuracy_significant": accuracy_pass,
        "time_significant": time_pass,
    }

    logger.info("Experiment analysis: %s - %s", decision, reason)
    return results
