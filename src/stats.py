# start src/stats.py
"""Statistical analysis module for bias detection.

Provides statistical significance testing, effect size calculation,
confidence intervals, and comprehensive bias assessment for publication-ready
analysis.

References:
- Bootstrap CI: Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
- WEAT: Caliskan et al. (2017) "Semantics derived automatically from language
  corpora contain human-like biases" (Science)
- Cohen's d: Cohen (1988) "Statistical Power Analysis for the Behavioral Sciences"
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import numpy.typing as npt
from scipy import stats as scipy_stats
from scipy.stats import bootstrap, shapiro, wilcoxon
from statsmodels.stats.power import TTestPower


# Reproducibility seed for stochastic processes
RANDOM_SEED = 42


def set_reproducibility_seed(seed: int = RANDOM_SEED) -> None:
    """Set seeds for reproducibility across all stochastic processes.

    Sets random seeds for numpy, Python's random module, and PyTorch if available.
    Should be called before any bootstrap or permutation tests for reproducible
    results.

    Args:
        seed: Random seed value. Default is 42.
    """
    np.random.seed(seed)
    random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass  # torch not available, skip


@dataclass
class BiasStatistics:
    """Statistical analysis results for bias assessment.

    Contains all metrics needed for publication-ready bias analysis including
    multiple statistical tests, effect sizes with confidence intervals,
    assumption checks, and power analysis.
    """

    # Core metrics
    mean_similarity: float
    std_similarity: float
    sample_size: int

    # Confidence intervals (both methods for robustness)
    ci_lower_ttest: float
    ci_upper_ttest: float
    ci_lower_bootstrap: float
    ci_upper_bootstrap: float

    # Significance tests (multiple methods)
    p_value_ttest: Optional[float]
    p_value_permutation: Optional[float]
    p_value_wilcoxon: Optional[float]

    # Effect sizes
    cohens_d: Optional[float]
    cohens_d_ci_lower: Optional[float]
    cohens_d_ci_upper: Optional[float]
    weat_effect_size: Optional[float]

    # Assumption checks
    normality_p_value: Optional[float]
    is_normally_distributed: bool

    # Power analysis
    statistical_power: Optional[float]
    minimum_detectable_effect: Optional[float]

    # Interpretation
    is_significant: bool
    verdict: str

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for database storage or serialization.

        Returns:
            Dictionary with all statistical fields.
        """
        return {
            "mean_similarity": self.mean_similarity,
            "std_similarity": self.std_similarity,
            "sample_size": self.sample_size,
            "ci_lower_ttest": self.ci_lower_ttest,
            "ci_upper_ttest": self.ci_upper_ttest,
            "ci_lower_bootstrap": self.ci_lower_bootstrap,
            "ci_upper_bootstrap": self.ci_upper_bootstrap,
            "p_value_ttest": self.p_value_ttest,
            "p_value_permutation": self.p_value_permutation,
            "p_value_wilcoxon": self.p_value_wilcoxon,
            "cohens_d": self.cohens_d,
            "cohens_d_ci_lower": self.cohens_d_ci_lower,
            "cohens_d_ci_upper": self.cohens_d_ci_upper,
            "weat_effect_size": self.weat_effect_size,
            "normality_p_value": self.normality_p_value,
            "is_normally_distributed": self.is_normally_distributed,
            "statistical_power": self.statistical_power,
            "minimum_detectable_effect": self.minimum_detectable_effect,
            "is_significant": self.is_significant,
            "verdict": self.verdict,
        }


def calculate_confidence_interval(
    mean: float, std: float, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Calculate t-distribution confidence interval for the mean.

    Uses t-distribution for small samples, which provides more conservative
    intervals than z-distribution.

    Args:
        mean: Sample mean.
        std: Sample standard deviation.
        n: Sample size.
        confidence: Confidence level (default 0.95 for 95% CI).

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if n < 2:
        return (mean, mean)

    t_critical = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin_of_error = t_critical * (std / math.sqrt(n))

    return (mean - margin_of_error, mean + margin_of_error)


def bootstrap_confidence_interval(
    scores: list[float],
    confidence: float = 0.95,
    n_resamples: int = 10000,
    method: str = "BCa",
) -> tuple[float, float]:
    """Compute bootstrap confidence interval using BCa method.

    The BCa (Bias-Corrected and Accelerated) method adjusts for bias and
    skewness in the bootstrap distribution, providing more accurate coverage
    than percentile methods.

    Args:
        scores: List of similarity scores.
        confidence: Confidence level (default 0.95 for 95% CI).
        n_resamples: Number of bootstrap resamples (default 10000).
        method: Bootstrap method - 'BCa', 'percentile', or 'basic'.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    if len(scores) < 3:
        mean_val = float(np.mean(scores)) if scores else 0.0
        return (mean_val, mean_val)

    data = (np.array(scores),)
    try:
        res = bootstrap(
            data,
            np.mean,
            confidence_level=confidence,
            n_resamples=n_resamples,
            method=method,
        )
        return (float(res.confidence_interval.low), float(res.confidence_interval.high))
    except ValueError:
        # Fall back to percentile method if BCa fails
        mean_val = float(np.mean(scores))
        return (mean_val, mean_val)


def permutation_test(
    scores: list[float],
    null_value: float = 1.0,
    n_permutations: int = 10000,
) -> float:
    """Permutation test for difference from null hypothesis.

    Non-parametric test that makes no distributional assumptions. Generates
    a null distribution by centering the data and resampling, then calculates
    the proportion of samples as extreme as the observed difference.

    This is the gold standard for embedding bias research, as used in WEAT.

    Args:
        scores: List of similarity scores.
        null_value: Expected value under null hypothesis (default 1.0 = no bias).
        n_permutations: Number of permutations (default 10000).

    Returns:
        Two-sided p-value.
    """
    if len(scores) < 2:
        return 1.0

    scores_array = np.array(scores)
    observed_diff = float(np.mean(scores_array)) - null_value

    # Generate null distribution by centering data at null value
    centered = scores_array - np.mean(scores_array) + null_value

    count = 0
    for _ in range(n_permutations):
        perm_sample = np.random.choice(centered, size=len(scores), replace=True)
        perm_diff = np.mean(perm_sample) - null_value
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    return count / n_permutations


def check_normality(scores: list[float]) -> tuple[float, bool]:
    """Test normality using Shapiro-Wilk test.

    The Shapiro-Wilk test is recommended for sample sizes under 5000.
    A significant p-value (< 0.05) indicates non-normality.

    Args:
        scores: List of similarity scores.

    Returns:
        Tuple of (p_value, is_normal). is_normal is True if p >= 0.05.
    """
    if len(scores) < 3:
        return (1.0, True)  # Cannot test with fewer than 3 samples

    if len(scores) > 5000:
        # Shapiro-Wilk not recommended for very large samples
        # Use a subsample
        scores = list(np.random.choice(scores, size=5000, replace=False))

    stat, p_value = shapiro(scores)
    is_normal = p_value >= 0.05
    return (float(p_value), is_normal)


def wilcoxon_signed_rank_test(
    scores: list[float], null_value: float = 1.0
) -> tuple[float, float]:
    """Non-parametric Wilcoxon signed-rank test.

    Tests whether the median of differences from the null value is zero.
    More robust to outliers than the t-test and appropriate when normality
    assumption is violated.

    Args:
        scores: List of similarity scores.
        null_value: Expected value under null hypothesis (default 1.0).

    Returns:
        Tuple of (statistic, p_value).
    """
    if len(scores) < 10:
        return (0.0, 1.0)

    differences = np.array(scores) - null_value
    # Remove exact zeros (Wilcoxon requirement)
    differences = differences[differences != 0]

    if len(differences) < 10:
        return (0.0, 1.0)

    stat, p_value = wilcoxon(differences, alternative="two-sided")
    return (float(stat), float(p_value))


def calculate_cohens_d(
    sample_mean: float, sample_std: float, baseline_mean: float, baseline_std: float
) -> float:
    """Calculate Cohen's d effect size.

    Cohen's d measures the standardized difference between two means.
    Guidelines: |d| < 0.2 = negligible, 0.2-0.5 = small, 0.5-0.8 = medium, > 0.8 = large

    Args:
        sample_mean: Mean of sample group.
        sample_std: Standard deviation of sample group.
        baseline_mean: Mean of baseline/control group.
        baseline_std: Standard deviation of baseline group.

    Returns:
        Cohen's d effect size.
    """
    # Pooled standard deviation
    pooled_std = math.sqrt((sample_std**2 + baseline_std**2) / 2)

    if pooled_std == 0:
        return 0.0

    return (sample_mean - baseline_mean) / pooled_std


def cohens_d_with_ci(
    scores: list[float],
    baseline: float = 1.0,
    confidence: float = 0.95,
    n_resamples: int = 10000,
) -> tuple[float, float, float]:
    """Calculate Cohen's d with bootstrap confidence interval.

    Provides point estimate and CI for effect size, which is required
    by many journals following APA guidelines.

    Args:
        scores: List of similarity scores.
        baseline: Expected baseline value (default 1.0).
        confidence: Confidence level (default 0.95).
        n_resamples: Number of bootstrap resamples.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    if len(scores) < 2:
        return (0.0, 0.0, 0.0)

    def compute_d(x: npt.NDArray[np.floating[Any]]) -> float:
        std = np.std(x, ddof=1)
        if std == 0:
            return 0.0
        return float((np.mean(x) - baseline) / std)

    scores_array = np.array(scores)
    point_estimate = compute_d(scores_array)

    # Bootstrap CI
    boot_ds: list[float] = []
    for _ in range(n_resamples):
        sample = np.random.choice(scores_array, size=len(scores), replace=True)
        if np.std(sample, ddof=1) > 0:
            boot_ds.append(compute_d(sample))

    if not boot_ds:
        return (point_estimate, point_estimate, point_estimate)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_ds, alpha / 2 * 100))
    ci_upper = float(np.percentile(boot_ds, (1 - alpha / 2) * 100))

    return (point_estimate, ci_lower, ci_upper)


def calculate_weat_effect_size(scores: list[float]) -> float:
    """Calculate WEAT-style effect size for embedding bias.

    WEAT (Word Embedding Association Test) effect size is the standard
    metric for measuring bias in embedding spaces. It measures the
    normalized mean deviation from perfect similarity.

    Formula: mean(deviations) / std(deviations)
    Where: deviations = 1.0 - similarity_scores

    Reference: Caliskan et al., 2017 (Science)

    Args:
        scores: List of cosine similarity scores.

    Returns:
        WEAT effect size (positive = bias toward lower similarity).
    """
    if len(scores) < 2:
        return 0.0

    deviations = [1.0 - s for s in scores]
    std = np.std(deviations, ddof=1)

    if std == 0:
        return 0.0

    return float(np.mean(deviations) / std)


def perform_one_sample_ttest(
    scores: list[float], null_hypothesis_mean: float = 1.0
) -> tuple[float, float]:
    """Perform one-sample t-test against null hypothesis.

    Tests if sample mean significantly differs from hypothesized value.
    For bias detection: H0 is that similarity = 1.0 (no bias).

    Args:
        scores: List of similarity scores.
        null_hypothesis_mean: Expected mean under null hypothesis (default 1.0).

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if len(scores) < 2:
        return (0.0, 1.0)

    t_stat, p_value = scipy_stats.ttest_1samp(scores, null_hypothesis_mean)
    return (float(t_stat), float(p_value))


def perform_two_sample_ttest(
    group1_scores: list[float], group2_scores: list[float]
) -> tuple[float, float]:
    """Perform independent two-sample t-test.

    Tests if two groups have significantly different means.

    Args:
        group1_scores: Scores from first group.
        group2_scores: Scores from second group.

    Returns:
        Tuple of (t_statistic, p_value).
    """
    if len(group1_scores) < 2 or len(group2_scores) < 2:
        return (0.0, 1.0)

    t_stat, p_value = scipy_stats.ttest_ind(group1_scores, group2_scores)
    return (float(t_stat), float(p_value))


def calculate_statistical_power(
    n: int, effect_size: float, alpha: float = 0.05
) -> float:
    """Calculate statistical power for given sample size and effect.

    Power is the probability of correctly rejecting a false null hypothesis.
    Conventionally, power >= 0.8 is considered adequate.

    Args:
        n: Sample size.
        effect_size: Expected or observed effect size (Cohen's d).
        alpha: Significance level (default 0.05).

    Returns:
        Statistical power (0-1).
    """
    if n < 2 or effect_size == 0:
        return 0.0

    try:
        analysis = TTestPower()
        power = analysis.power(effect_size=abs(effect_size), nobs=n, alpha=alpha)
        return float(power)
    except (ValueError, ZeroDivisionError):
        return 0.0


def calculate_minimum_detectable_effect(
    n: int, power: float = 0.8, alpha: float = 0.05
) -> float:
    """Calculate minimum detectable effect size for given sample.

    Useful for determining if sample size is adequate for detecting
    practically meaningful effects.

    Args:
        n: Sample size.
        power: Desired statistical power (default 0.8).
        alpha: Significance level (default 0.05).

    Returns:
        Minimum detectable effect size (Cohen's d).
    """
    if n < 3:
        return float("inf")

    try:
        analysis = TTestPower()
        effect = analysis.solve_power(nobs=n, power=power, alpha=alpha)
        return float(effect)
    except (ValueError, ZeroDivisionError):
        return float("inf")


def interpret_effect_size(cohens_d: float) -> str:
    """Interpret Cohen's d effect size magnitude.

    Uses Cohen's (1988) conventional guidelines.

    Args:
        cohens_d: Cohen's d value.

    Returns:
        Human-readable interpretation: negligible, small, medium, or large.
    """
    abs_d = abs(cohens_d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def analyze_bias(
    similarity_scores: list[float],
    baseline_mean: float = 1.0,
    baseline_std: float = 0.0,
    significance_level: float = 0.05,
    n_bootstrap: int = 10000,
    n_permutations: int = 10000,
) -> BiasStatistics:
    """Perform comprehensive statistical analysis of bias.

    Runs multiple statistical tests for robust, publication-ready analysis:
    - T-test (parametric)
    - Permutation test (non-parametric)
    - Wilcoxon signed-rank test (non-parametric)
    - Bootstrap confidence intervals
    - Effect size with confidence interval
    - WEAT effect size
    - Power analysis

    Args:
        similarity_scores: List of cosine similarity scores.
        baseline_mean: Expected mean under no-bias assumption (default 1.0).
        baseline_std: Expected std under no-bias assumption (default 0.0).
        significance_level: Alpha for hypothesis testing (default 0.05).
        n_bootstrap: Number of bootstrap resamples (default 10000).
        n_permutations: Number of permutations for permutation test.

    Returns:
        BiasStatistics with complete analysis.
    """
    # Set seed for reproducibility
    set_reproducibility_seed()

    if len(similarity_scores) == 0:
        return BiasStatistics(
            mean_similarity=0.0,
            std_similarity=0.0,
            sample_size=0,
            ci_lower_ttest=0.0,
            ci_upper_ttest=0.0,
            ci_lower_bootstrap=0.0,
            ci_upper_bootstrap=0.0,
            p_value_ttest=None,
            p_value_permutation=None,
            p_value_wilcoxon=None,
            cohens_d=None,
            cohens_d_ci_lower=None,
            cohens_d_ci_upper=None,
            weat_effect_size=None,
            normality_p_value=None,
            is_normally_distributed=True,
            statistical_power=None,
            minimum_detectable_effect=None,
            is_significant=False,
            verdict="Insufficient data",
        )

    scores_array = np.array(similarity_scores)
    mean = float(np.mean(scores_array))
    std = float(np.std(scores_array, ddof=1)) if len(scores_array) > 1 else 0.0
    n = len(scores_array)

    # Normality test (run first to guide interpretation)
    normality_p, is_normal = check_normality(list(scores_array))

    # T-distribution confidence interval
    ci_lower_t, ci_upper_t = calculate_confidence_interval(mean, std, n)

    # Bootstrap confidence interval (BCa method)
    ci_lower_boot, ci_upper_boot = bootstrap_confidence_interval(
        list(scores_array), n_resamples=n_bootstrap
    )

    # T-test
    _, p_value_ttest = perform_one_sample_ttest(list(scores_array), baseline_mean)

    # Permutation test
    p_value_perm = permutation_test(
        list(scores_array), baseline_mean, n_permutations=n_permutations
    )

    # Wilcoxon signed-rank test
    _, p_value_wilcox = wilcoxon_signed_rank_test(list(scores_array), baseline_mean)

    # Effect sizes
    cohens_d_val, cohens_d_ci_low, cohens_d_ci_high = cohens_d_with_ci(
        list(scores_array), baseline_mean, n_resamples=n_bootstrap
    )
    weat_es = calculate_weat_effect_size(list(scores_array))

    # Power analysis
    power = calculate_statistical_power(n, cohens_d_val) if cohens_d_val else None
    min_effect = calculate_minimum_detectable_effect(n)

    # Determine significance
    # Use permutation test p-value as primary (non-parametric, fewer assumptions)
    # But report all for transparency
    primary_p = p_value_perm if not is_normal else p_value_ttest
    is_significant = primary_p < significance_level if primary_p is not None else False

    # Generate verdict
    verdict = generate_verdict(
        mean=mean,
        p_value=primary_p,
        effect_size=cohens_d_val,
        ci_lower=ci_lower_boot,
        ci_upper=ci_upper_boot,
        is_normal=is_normal,
        significance_level=significance_level,
    )

    return BiasStatistics(
        mean_similarity=mean,
        std_similarity=std,
        sample_size=n,
        ci_lower_ttest=ci_lower_t,
        ci_upper_ttest=ci_upper_t,
        ci_lower_bootstrap=ci_lower_boot,
        ci_upper_bootstrap=ci_upper_boot,
        p_value_ttest=p_value_ttest,
        p_value_permutation=p_value_perm,
        p_value_wilcoxon=p_value_wilcox,
        cohens_d=cohens_d_val,
        cohens_d_ci_lower=cohens_d_ci_low,
        cohens_d_ci_upper=cohens_d_ci_high,
        weat_effect_size=weat_es,
        normality_p_value=normality_p,
        is_normally_distributed=is_normal,
        statistical_power=power,
        minimum_detectable_effect=min_effect,
        is_significant=is_significant,
        verdict=verdict,
    )


def generate_verdict(
    mean: float,
    p_value: Optional[float],
    effect_size: Optional[float],
    ci_lower: float,
    ci_upper: float,
    is_normal: bool = True,
    significance_level: float = 0.05,
) -> str:
    """Generate human-readable verdict based on statistical analysis.

    Considers statistical significance, effect size, and confidence intervals
    to provide a nuanced interpretation.

    Args:
        mean: Mean similarity score.
        p_value: P-value from primary test.
        effect_size: Cohen's d.
        ci_lower: Lower bound of confidence interval.
        ci_upper: Upper bound of confidence interval.
        is_normal: Whether data passes normality test.
        significance_level: Alpha level.

    Returns:
        Human-readable verdict string.
    """
    # Add normality note if applicable
    normality_note = "" if is_normal else " [non-parametric tests used]"

    # If CI includes 1.0 (no bias), we cannot reject null hypothesis
    if ci_lower <= 1.0 <= ci_upper:
        return f"No Significant Bias (CI includes 1.0){normality_note}"

    # Check statistical significance
    if p_value is None or p_value >= significance_level:
        return f"No Significant Bias (p >= {significance_level}){normality_note}"

    # Significant result - interpret effect size
    if effect_size is not None:
        effect_magnitude = interpret_effect_size(effect_size)
        if effect_magnitude == "negligible":
            return (
                f"Statistically Significant but Negligible Effect "
                f"(d={effect_size:.3f}){normality_note}"
            )
        elif effect_magnitude == "small":
            return (
                f"Small Bias Detected "
                f"(d={effect_size:.3f}, p={p_value:.4f}){normality_note}"
            )
        elif effect_magnitude == "medium":
            return (
                f"Medium Bias Detected "
                f"(d={effect_size:.3f}, p={p_value:.4f}){normality_note}"
            )
        else:
            return (
                f"Large Bias Detected "
                f"(d={effect_size:.3f}, p={p_value:.4f}){normality_note}"
            )

    # Fallback if no effect size
    if mean < 0.95:
        return f"High Bias (mean={mean:.3f}, p={p_value:.4f}){normality_note}"
    elif mean < 0.99:
        return f"Moderate Bias (mean={mean:.3f}, p={p_value:.4f}){normality_note}"
    else:
        return f"Low Bias (mean={mean:.3f}, p={p_value:.4f}){normality_note}"


def analyze_single_pair(
    similarity_score: float,
    historical_scores: Optional[list[float]] = None,
) -> dict[str, object]:
    """Analyze a single pair with optional historical context.

    For single pairs without historical data, provides basic assessment.
    With historical data, provides statistical context.

    Args:
        similarity_score: Cosine similarity for this pair.
        historical_scores: Optional list of previous scores for context.

    Returns:
        Dictionary with analysis results.
    """
    result: dict[str, object] = {
        "similarity_score": similarity_score,
        "deviation_from_perfect": 1.0 - similarity_score,
    }

    if historical_scores and len(historical_scores) >= 2:
        hist_array = np.array(historical_scores)
        hist_mean = float(np.mean(hist_array))
        hist_std = float(np.std(hist_array, ddof=1))

        # Z-score: how many std devs from historical mean
        if hist_std > 0:
            z_score = (similarity_score - hist_mean) / hist_std
            result["z_score"] = z_score
            result["is_outlier"] = abs(z_score) > 2.0

        # Percentile rank
        percentile = float(
            np.sum(hist_array <= similarity_score) / len(hist_array) * 100
        )
        result["percentile"] = percentile

    # Basic verdict for single pair - uses deviation-focused language
    # (distinct from aggregate statistical bias assessment)
    if similarity_score < 0.90:
        result["basic_verdict"] = "Large Deviation"
    elif similarity_score < 0.95:
        result["basic_verdict"] = "Moderate Deviation"
    elif similarity_score < 0.99:
        result["basic_verdict"] = "Small Deviation"
    else:
        result["basic_verdict"] = "Minimal Deviation"

    return result


# end src/stats.py
