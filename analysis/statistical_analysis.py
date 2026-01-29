#!/usr/bin/env python3
"""
Statistical Analysis for Genre-Mimicry Paper
============================================
Analyzes disclaimer rates across models and genres to test the "Violence Gap" hypothesis.

Author: Farzulla Research
Date: January 2026
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# For mixed-effects models
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    from statsmodels.regression.mixed_linear_model import MixedLM
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False
    print("[WARNING] statsmodels not available - mixed-effects models will be skipped")


# ============================================================================
# Data Loading
# ============================================================================

def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file and return list of records."""
    records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def load_all_data(data_dir: Path) -> pd.DataFrame:
    """Load all JSONL files and combine into single DataFrame."""
    all_records = []

    # Define model mappings (filename -> clean name, is_abliterated, is_control)
    model_info = {
        'genre_mimicry_results.jsonl': ('Qwen2.5-32B-Abl', True, False),
        'genre_mimicry_results_qwen2.5_32b_abliterated.jsonl': ('Qwen2.5-32B-Abl-2', True, False),
        'genre_mimicry_results_qwen3_30b.jsonl': ('Qwen3-30B', False, True),  # Control
        'genre_mimicry_results_qwen3_8b_abliterated.jsonl': ('Qwen3-8B-Abl', True, False),
        'genre_mimicry_results_gemma-3-27b-it-abliterated-normpreserve-i1.jsonl': ('Gemma3-27B-Abl', True, False),
        'genre_mimicry_results_qwen3-vl-8b-instruct-abliterated-v2.0.jsonl': ('Qwen3-VL-8B-Abl', True, False),
        'genre_mimicry_results_llama-3.2-8x3b-moe-dark-champion-instruct-uncensored-abliterated-18.4b.jsonl': ('Llama-MoE-18B-Abl', True, False),
        'genre_mimicry_results_mistralai_devstral-small-2-2512.jsonl': ('Devstral-Small', False, True),  # Control
        'genre_mimicry_results_openai-gpt-oss-20b-abliterated-uncensored-neo-imatrix.jsonl': ('GPT-OSS-20B-Abl', True, False),
    }

    for filename, (model_name, is_abliterated, is_control) in model_info.items():
        filepath = data_dir / filename
        if filepath.exists():
            records = load_jsonl(filepath)
            for r in records:
                r['model_clean'] = model_name
                r['is_abliterated'] = is_abliterated
                r['is_control'] = is_control
                # Override model field if not present
                if 'model' not in r:
                    r['model'] = model_name
            all_records.extend(records)
            print(f"  Loaded {len(records):3d} records from {filename}")
        else:
            print(f"  [SKIP] {filename} not found")

    df = pd.DataFrame(all_records)

    # Create genre category (high-level grouping)
    def get_genre(cat):
        if 'InfoSec' in cat or 'Phishing' in cat:
            return 'InfoSec'
        elif 'Violence' in cat or 'Murder' in cat:
            return 'Violence'
        elif 'Chemistry' in cat or 'Explosive' in cat:
            return 'Chemistry'
        else:
            return cat

    df['genre'] = df['category'].apply(get_genre)

    # Convert has_disclaimer to int for easier stats
    df['disclaimer_int'] = df['has_disclaimer'].astype(int)

    return df


# ============================================================================
# Statistical Tests
# ============================================================================

def cramers_v(contingency_table):
    """Calculate Cramer's V for effect size."""
    chi2 = stats.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    min_dim = min(contingency_table.shape) - 1
    if min_dim == 0 or n == 0:
        return 0.0
    return np.sqrt(chi2 / (n * min_dim))


def odds_ratio(a, b, c, d):
    """
    Calculate odds ratio for 2x2 table.

         | Disclaimer | No Disclaimer
    -----|------------|---------------
    Row1 |     a      |      b
    Row2 |     c      |      d

    OR = (a*d) / (b*c)
    """
    if b * c == 0:
        return np.inf if a * d > 0 else np.nan
    return (a * d) / (b * c)


def odds_ratio_ci(a, b, c, d, alpha=0.05):
    """Calculate odds ratio with 95% CI using log method."""
    or_val = odds_ratio(a, b, c, d)
    if np.isinf(or_val) or np.isnan(or_val):
        return or_val, (np.nan, np.nan)

    # Log odds ratio SE
    with np.errstate(divide='ignore'):
        log_or = np.log(or_val)
        se = np.sqrt(1/max(a,0.5) + 1/max(b,0.5) + 1/max(c,0.5) + 1/max(d,0.5))

    z = stats.norm.ppf(1 - alpha/2)
    ci_low = np.exp(log_or - z * se)
    ci_high = np.exp(log_or + z * se)

    return or_val, (ci_low, ci_high)


def run_chi_square_test(df, group_col='genre', outcome_col='has_disclaimer'):
    """Run chi-square test for independence."""
    contingency = pd.crosstab(df[group_col], df[outcome_col])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    cramer_v = cramers_v(contingency)

    return {
        'test': 'Chi-Square',
        'chi2': chi2,
        'p_value': p_value,
        'dof': dof,
        'cramers_v': cramer_v,
        'contingency_table': contingency,
        'expected': expected
    }


def run_fisher_exact(df, genre1, genre2, outcome_col='has_disclaimer'):
    """Run Fisher's exact test comparing two genres."""
    df_subset = df[df['genre'].isin([genre1, genre2])]

    if len(df_subset) == 0:
        return None

    # Build 2x2 table manually to avoid indexing issues
    g1_disclaimer = len(df_subset[(df_subset['genre'] == genre1) & (df_subset[outcome_col] == True)])
    g1_no_disclaimer = len(df_subset[(df_subset['genre'] == genre1) & (df_subset[outcome_col] == False)])
    g2_disclaimer = len(df_subset[(df_subset['genre'] == genre2) & (df_subset[outcome_col] == True)])
    g2_no_disclaimer = len(df_subset[(df_subset['genre'] == genre2) & (df_subset[outcome_col] == False)])

    # Build table: [[g1_disc, g1_no_disc], [g2_disc, g2_no_disc]]
    table = np.array([[g1_disclaimer, g1_no_disclaimer],
                      [g2_disclaimer, g2_no_disclaimer]])

    # Check for valid table
    if table.sum() == 0:
        return None

    # Fisher's exact test
    odds_r, p_value = stats.fisher_exact(table)

    # Get cell values for OR calculation with CI
    a, b = table[0]
    c, d = table[1]
    or_val, ci = odds_ratio_ci(a, b, c, d)

    return {
        'test': 'Fisher Exact',
        'comparison': f'{genre1} vs {genre2}',
        'odds_ratio': odds_r,
        'odds_ratio_ci': ci,
        'p_value': p_value,
        'table': table,
        'genre1_disclaimer_rate': a / (a + b) if (a + b) > 0 else 0,
        'genre2_disclaimer_rate': c / (c + d) if (c + d) > 0 else 0,
    }


def run_proportion_ztest(n1, p1, n2, p2):
    """Two-proportion z-test."""
    from statsmodels.stats.proportion import proportions_ztest

    count = np.array([int(n1 * p1), int(n2 * p2)])
    nobs = np.array([n1, n2])

    stat, p_value = proportions_ztest(count, nobs, alternative='two-sided')
    return {'z_stat': stat, 'p_value': p_value}


def run_mixed_effects_logistic(df):
    """
    Run mixed-effects models with model as random effect.

    Model: disclaimer ~ genre + (1|model)

    NOTE: statsmodels does not have a straightforward mixed-effects logistic
    regression (glmer equivalent). We fit:
    1. Linear Mixed Model (LPM approximation) - fast, interpretable on probability scale
    2. GEE with logit link - population-averaged effects on log-odds scale
    3. Standard logistic for comparison

    This addresses reviewer concerns about non-independence of observations
    within models by accounting for model-level variance.
    """
    if not HAS_STATSMODELS:
        return {
            'error': 'statsmodels not available',
            'model_type': 'mixed_effects_logistic'
        }

    results = {}

    # Prepare data - need clean categorical coding
    df_model = df.copy()

    # Create binary outcome
    df_model['disclaimer_binary'] = df_model['has_disclaimer'].astype(int)

    # Create dummy variables for genre with Violence as reference
    # This makes interpretation clearer for the Violence Gap hypothesis
    df_model['is_infosec'] = (df_model['genre'] == 'InfoSec').astype(int)
    df_model['is_chemistry'] = (df_model['genre'] == 'Chemistry').astype(int)
    df_model['is_finance'] = (df_model['genre'] == 'Finance_Fraud').astype(int)
    # Violence is reference category (omitted)

    try:
        # Method 1: Mixed-effects model using MixedLM
        # Note: statsmodels doesn't have a true mixed-effects logistic regression,
        # so we use linear mixed model on binary outcome (approximation)
        # This is a common approach when glmer is not available

        formula = "disclaimer_binary ~ is_infosec + is_chemistry + is_finance"

        model = smf.mixedlm(
            formula,
            data=df_model,
            groups=df_model['model_clean'],
            re_formula="~1"  # Random intercept only
        )

        fit = model.fit(method='powell', maxiter=500)

        results['mixed_effects_linear'] = {
            'model_type': 'Linear Probability Model with Random Intercepts (LPM-RE)',
            'note': 'Coefficients are on probability scale (0-1), not log-odds. This is a linear approximation; see gee_logistic for proper log-odds inference.',
            'formula': 'disclaimer ~ genre + (1|model)',
            'reference_category': 'Violence',
            'n_observations': int(fit.nobs),
            'n_groups': int(fit.k_fe + fit.k_re),
            'converged': fit.converged,
            'log_likelihood': float(fit.llf) if hasattr(fit, 'llf') else None,
            'aic': float(fit.aic) if hasattr(fit, 'aic') else None,
            'bic': float(fit.bic) if hasattr(fit, 'bic') else None,
            'fixed_effects': {},
            'random_effects_variance': float(fit.cov_re.iloc[0, 0]) if hasattr(fit, 'cov_re') else None,
            'icc': None  # Will calculate below
        }

        # Extract fixed effects
        for param in fit.params.index:
            coef = float(fit.params[param])
            se = float(fit.bse[param]) if param in fit.bse.index else None
            pval = float(fit.pvalues[param]) if param in fit.pvalues.index else None
            ci_low, ci_high = fit.conf_int().loc[param] if param in fit.conf_int().index else (None, None)

            results['mixed_effects_linear']['fixed_effects'][param] = {
                'coefficient': coef,
                'std_error': se,
                'p_value': pval,
                'ci_95': [float(ci_low) if ci_low is not None else None,
                          float(ci_high) if ci_high is not None else None]
            }

        # Calculate ICC (Intraclass Correlation Coefficient)
        # ICC = variance_between / (variance_between + variance_within)
        # For binary outcomes with linear approximation, residual variance ≈ p(1-p)
        var_between = float(fit.cov_re.iloc[0, 0])
        p_hat = df_model['disclaimer_binary'].mean()
        var_within_approx = p_hat * (1 - p_hat)
        icc = var_between / (var_between + var_within_approx)
        results['mixed_effects_linear']['icc'] = float(icc)
        results['mixed_effects_linear']['icc_interpretation'] = (
            f"ICC = {icc:.3f}: {icc*100:.1f}% of variance in disclaimer presence is "
            f"attributable to between-model differences"
        )

        # Genre effect summary for easy interpretation
        results['mixed_effects_linear']['genre_effects_summary'] = {
            'interpretation': 'Coefficients show change in disclaimer probability relative to Violence category',
            'violence_baseline': float(fit.params['Intercept']),
            'infosec_vs_violence': {
                'difference': float(fit.params.get('is_infosec', 0)),
                'p_value': float(fit.pvalues.get('is_infosec', np.nan)),
                'significant': bool(fit.pvalues.get('is_infosec', 1) < 0.05)
            },
            'chemistry_vs_violence': {
                'difference': float(fit.params.get('is_chemistry', 0)),
                'p_value': float(fit.pvalues.get('is_chemistry', np.nan)),
                'significant': bool(fit.pvalues.get('is_chemistry', 1) < 0.05)
            },
            'finance_vs_violence': {
                'difference': float(fit.params.get('is_finance', 0)),
                'p_value': float(fit.pvalues.get('is_finance', np.nan)),
                'significant': bool(fit.pvalues.get('is_finance', 1) < 0.05)
            }
        }

        print("\n  Mixed-effects model fitted successfully")
        print(f"    ICC: {icc:.3f} ({icc*100:.1f}% variance between models)")
        print(f"    Violence baseline: {fit.params['Intercept']:.3f}")
        for genre_var in ['is_infosec', 'is_chemistry', 'is_finance']:
            if genre_var in fit.params:
                sig = "*" if fit.pvalues[genre_var] < 0.05 else ""
                sig = "**" if fit.pvalues[genre_var] < 0.01 else sig
                sig = "***" if fit.pvalues[genre_var] < 0.001 else sig
                print(f"    {genre_var}: {fit.params[genre_var]:+.3f} (p={fit.pvalues[genre_var]:.4f}) {sig}")

    except Exception as e:
        results['mixed_effects_linear'] = {
            'error': str(e),
            'model_type': 'Mixed Linear Model'
        }
        print(f"\n  [ERROR] Mixed-effects model failed: {e}")

    # Method 2: GEE with logit link (cluster-robust logistic regression)
    try:
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod.families import Binomial
        from statsmodels.genmod.cov_struct import Exchangeable

        # Prepare data with numeric group codes
        df_gee = df_model.copy()
        df_gee['model_code'] = pd.Categorical(df_gee['model_clean']).codes

        # Sort by group for GEE
        df_gee = df_gee.sort_values('model_code')

        formula_gee = "disclaimer_binary ~ is_infosec + is_chemistry + is_finance"
        gee_model = GEE.from_formula(
            formula_gee,
            groups='model_code',
            data=df_gee,
            family=Binomial(),
            cov_struct=Exchangeable()
        )
        gee_fit = gee_model.fit()

        results['gee_logistic'] = {
            'model_type': 'GEE Logistic (exchangeable correlation, cluster-robust SEs)',
            'note': 'Log-odds coefficients; accounts for within-model correlation',
            'formula': 'disclaimer ~ genre + cluster(model)',
            'reference_category': 'Violence',
            'n_observations': int(gee_fit.nobs),
            'n_clusters': int(df_gee['model_code'].nunique()),
            'fixed_effects': {}
        }

        for param in gee_fit.params.index:
            coef = float(gee_fit.params[param])
            se = float(gee_fit.bse[param])
            pval = float(gee_fit.pvalues[param])
            ci_low, ci_high = gee_fit.conf_int().loc[param]
            odds_ratio = np.exp(coef)
            or_ci = (np.exp(ci_low), np.exp(ci_high))

            results['gee_logistic']['fixed_effects'][param] = {
                'coefficient_log_odds': coef,
                'std_error': se,
                'p_value': pval,
                'ci_95_log_odds': [float(ci_low), float(ci_high)],
                'odds_ratio': float(odds_ratio),
                'or_ci_95': [float(or_ci[0]), float(or_ci[1])]
            }

        print("\n  GEE logistic (cluster-robust) fitted successfully")
        print(f"    Violence (ref) log-odds: {gee_fit.params['Intercept']:.3f}")
        for genre_var in ['is_infosec', 'is_chemistry', 'is_finance']:
            if genre_var in gee_fit.params:
                or_val = np.exp(gee_fit.params[genre_var])
                sig = "*" if gee_fit.pvalues[genre_var] < 0.05 else ""
                sig = "**" if gee_fit.pvalues[genre_var] < 0.01 else sig
                sig = "***" if gee_fit.pvalues[genre_var] < 0.001 else sig
                print(f"    {genre_var}: OR={or_val:.2f} (p={gee_fit.pvalues[genre_var]:.4f}) {sig}")

    except Exception as e:
        results['gee_logistic'] = {
            'error': str(e),
            'model_type': 'GEE Logistic'
        }
        print(f"\n  [WARNING] GEE logistic failed: {e}")

    # Method 3: Standard logistic regression for comparison
    try:
        from statsmodels.discrete.discrete_model import Logit

        X = df_model[['is_infosec', 'is_chemistry', 'is_finance']]
        X = sm.add_constant(X)
        y = df_model['disclaimer_binary']

        logit_model = Logit(y, X)
        logit_fit = logit_model.fit(disp=0)

        results['standard_logistic'] = {
            'model_type': 'Standard Logistic Regression (no clustering)',
            'note': 'Does NOT account for model-level clustering; provided for comparison',
            'formula': 'disclaimer ~ genre',
            'reference_category': 'Violence',
            'n_observations': int(logit_fit.nobs),
            'pseudo_r2': float(logit_fit.prsquared),
            'log_likelihood': float(logit_fit.llf),
            'aic': float(logit_fit.aic),
            'bic': float(logit_fit.bic),
            'fixed_effects': {}
        }

        for param in logit_fit.params.index:
            coef = float(logit_fit.params[param])
            se = float(logit_fit.bse[param])
            pval = float(logit_fit.pvalues[param])
            ci_low, ci_high = logit_fit.conf_int().loc[param]
            odds_ratio = np.exp(coef)
            or_ci = (np.exp(ci_low), np.exp(ci_high))

            results['standard_logistic']['fixed_effects'][param] = {
                'coefficient_log_odds': coef,
                'std_error': se,
                'p_value': pval,
                'ci_95_log_odds': [float(ci_low), float(ci_high)],
                'odds_ratio': float(odds_ratio),
                'or_ci_95': [float(or_ci[0]), float(or_ci[1])]
            }

        print("\n  Standard logistic (no clustering) fitted for comparison")

    except Exception as e:
        results['standard_logistic'] = {
            'error': str(e),
            'model_type': 'Standard Logistic Regression'
        }

    return results


# ============================================================================
# Analysis Functions
# ============================================================================

def analyze_overall_rates(df):
    """Compute overall disclaimer rates by genre and model."""
    results = {}

    # By genre
    genre_rates = df.groupby('genre').agg(
        n=('has_disclaimer', 'count'),
        disclaimers=('disclaimer_int', 'sum'),
        rate=('disclaimer_int', 'mean')
    ).round(4)
    results['by_genre'] = genre_rates

    # By model
    model_rates = df.groupby('model_clean').agg(
        n=('has_disclaimer', 'count'),
        disclaimers=('disclaimer_int', 'sum'),
        rate=('disclaimer_int', 'mean'),
        is_abliterated=('is_abliterated', 'first')
    ).round(4)
    results['by_model'] = model_rates

    # By model and genre
    cross_rates = df.groupby(['model_clean', 'genre']).agg(
        n=('has_disclaimer', 'count'),
        disclaimers=('disclaimer_int', 'sum'),
        rate=('disclaimer_int', 'mean')
    ).round(4)
    results['by_model_genre'] = cross_rates

    return results


def test_violence_gap(df):
    """
    Test the Violence Gap hypothesis:
    Is violence disclaimer rate significantly lower than other genres?
    """
    results = {}

    # Overall comparison: Violence vs Non-Violence
    df_copy = df.copy()
    df_copy['is_violence'] = df_copy['genre'] == 'Violence'

    violence_data = df_copy[df_copy['is_violence']]
    other_data = df_copy[~df_copy['is_violence']]

    v_n = len(violence_data)
    v_rate = violence_data['disclaimer_int'].mean()
    o_n = len(other_data)
    o_rate = other_data['disclaimer_int'].mean()

    results['violence_rate'] = v_rate
    results['other_rate'] = o_rate
    results['violence_n'] = v_n
    results['other_n'] = o_n
    results['rate_difference'] = o_rate - v_rate  # Positive = Violence has LOWER rate

    # Chi-square test: Violence vs Non-Violence
    contingency = pd.crosstab(df_copy['is_violence'], df_copy['has_disclaimer'])
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    results['chi2'] = chi2
    results['p_value'] = p_value
    results['cramers_v'] = cramers_v(contingency)

    # Fisher's exact for robustness
    table = contingency.values
    fisher_or, fisher_p = stats.fisher_exact(table)
    results['fisher_or'] = fisher_or
    results['fisher_p'] = fisher_p

    # Odds ratio with CI
    a, b = contingency.loc[True].values  # Violence: [disclaimer, no_disclaimer]
    c, d = contingency.loc[False].values  # Other: [disclaimer, no_disclaimer]
    or_val, ci = odds_ratio_ci(a, b, c, d)
    results['odds_ratio'] = or_val
    results['odds_ratio_ci'] = ci

    return results


def test_violence_gap_per_model(df):
    """Test Violence Gap hypothesis for each model."""
    per_model_results = {}

    for model in df['model_clean'].unique():
        model_df = df[df['model_clean'] == model]

        violence_data = model_df[model_df['genre'] == 'Violence']
        other_data = model_df[model_df['genre'] != 'Violence']

        if len(violence_data) == 0 or len(other_data) == 0:
            continue

        v_rate = violence_data['disclaimer_int'].mean()
        o_rate = other_data['disclaimer_int'].mean()

        # Create contingency table
        is_violence = model_df['genre'] == 'Violence'
        contingency = pd.crosstab(is_violence, model_df['has_disclaimer'])

        if contingency.shape == (2, 2):
            fisher_or, fisher_p = stats.fisher_exact(contingency.values)
        else:
            fisher_or, fisher_p = np.nan, np.nan

        per_model_results[model] = {
            'violence_rate': v_rate,
            'violence_n': len(violence_data),
            'other_rate': o_rate,
            'other_n': len(other_data),
            'gap': o_rate - v_rate,
            'fisher_or': fisher_or,
            'fisher_p': fisher_p,
            'is_abliterated': model_df['is_abliterated'].iloc[0]
        }

    return per_model_results


def holm_correction(pvalues):
    """
    Apply Holm-Bonferroni correction for multiple testing.

    Returns adjusted p-values maintaining the original order.
    """
    n = len(pvalues)
    if n == 0:
        return []

    # Sort p-values and track original indices
    indexed = sorted(enumerate(pvalues), key=lambda x: x[1])
    adjusted = [0.0] * n

    # Apply Holm correction: p_adj[i] = min(1, p[i] * (n - rank + 1))
    # with monotonicity enforced
    max_adj = 0.0
    for rank, (orig_idx, pval) in enumerate(indexed):
        adj_p = pval * (n - rank)
        adj_p = min(adj_p, 1.0)
        adj_p = max(adj_p, max_adj)  # Enforce monotonicity
        max_adj = adj_p
        adjusted[orig_idx] = adj_p

    return adjusted


def pairwise_genre_comparisons(df):
    """Run pairwise Fisher's exact tests between all genres with Holm correction."""
    genres = df['genre'].unique()
    comparisons = []

    for i, g1 in enumerate(genres):
        for g2 in genres[i+1:]:
            result = run_fisher_exact(df, g1, g2)
            if result:
                comparisons.append(result)

    # Apply Holm correction for multiple testing
    if comparisons:
        pvalues = [c['p_value'] for c in comparisons]
        adjusted = holm_correction(pvalues)
        for i, comp in enumerate(comparisons):
            comp['p_value_holm'] = adjusted[i]
            comp['significant_holm'] = adjusted[i] < 0.05

    return comparisons


def analyze_warn_and_answer(harm_scores_path: Path):
    """
    Analyze the "warn-and-answer" pattern: responses with disclaimers that
    still contain harmful content per Llama Guard classification.

    Creates 2x2 contingency table:
        has_disclaimer × guard_safe

    The key metric is: P(unsafe | disclaimer) - how often models give warnings
    but still produce harmful content (the "decorative disclaimer" pattern).
    """
    if not harm_scores_path.exists():
        return {'error': f'File not found: {harm_scores_path}'}

    # Load Llama Guard scored data
    records = []
    with open(harm_scores_path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if not records:
        return {'error': 'No records in harm scores file'}

    df = pd.DataFrame(records)

    # Ensure required columns exist
    if 'has_disclaimer' not in df.columns or 'guard_safe' not in df.columns:
        return {'error': 'Missing required columns: has_disclaimer, guard_safe'}

    # Extract genre from category if not present
    if 'genre' not in df.columns and 'category' in df.columns:
        def get_genre(cat):
            if 'InfoSec' in str(cat) or 'Phishing' in str(cat):
                return 'InfoSec'
            elif 'Violence' in str(cat) or 'Murder' in str(cat):
                return 'Violence'
            elif 'Chemistry' in str(cat) or 'Explosive' in str(cat):
                return 'Chemistry'
            elif 'Finance' in str(cat) or 'Fraud' in str(cat):
                return 'Finance_Fraud'
            else:
                return str(cat)
        df['genre'] = df['category'].apply(get_genre) if 'category' in df.columns else df.get('genre', 'Unknown')

    results = {
        'n_total': len(df),
        'n_with_disclaimer': int(df['has_disclaimer'].sum()),
        'n_guard_unsafe': int((~df['guard_safe']).sum()),
    }

    # Create 2x2 contingency table
    # Rows: has_disclaimer (True/False)
    # Cols: guard_safe (True/False)
    crosstab = pd.crosstab(
        df['has_disclaimer'],
        df['guard_safe'],
        margins=True
    )

    # Extract quadrant counts
    # Format: [disclaimer_present][content_safe]
    try:
        disclaim_safe = crosstab.loc[True, True] if True in crosstab.index and True in crosstab.columns else 0
        disclaim_unsafe = crosstab.loc[True, False] if True in crosstab.index and False in crosstab.columns else 0
        no_disclaim_safe = crosstab.loc[False, True] if False in crosstab.index and True in crosstab.columns else 0
        no_disclaim_unsafe = crosstab.loc[False, False] if False in crosstab.index and False in crosstab.columns else 0
    except:
        disclaim_safe = disclaim_unsafe = no_disclaim_safe = no_disclaim_unsafe = 0

    results['contingency'] = {
        'disclaimer_safe': int(disclaim_safe),
        'disclaimer_unsafe': int(disclaim_unsafe),  # The "warn-and-answer" pattern
        'no_disclaimer_safe': int(no_disclaim_safe),
        'no_disclaimer_unsafe': int(no_disclaim_unsafe)
    }

    # Key metrics
    total_with_disclaimer = disclaim_safe + disclaim_unsafe
    total_without_disclaimer = no_disclaim_safe + no_disclaim_unsafe

    results['warn_and_answer'] = {
        'count': int(disclaim_unsafe),
        'rate_given_disclaimer': float(disclaim_unsafe / total_with_disclaimer) if total_with_disclaimer > 0 else 0.0,
        'interpretation': 'Proportion of responses with disclaimers that still contain harmful content'
    }

    results['silent_harm'] = {
        'count': int(no_disclaim_unsafe),
        'rate_given_no_disclaimer': float(no_disclaim_unsafe / total_without_disclaimer) if total_without_disclaimer > 0 else 0.0,
        'interpretation': 'Proportion of responses without disclaimers that contain harmful content'
    }

    # Per-genre breakdown
    genre_breakdown = {}
    for genre in df['genre'].unique():
        g_df = df[df['genre'] == genre]
        g_with_disclaim = g_df[g_df['has_disclaimer'] == True]
        g_disclaim_unsafe = len(g_with_disclaim[g_with_disclaim['guard_safe'] == False])
        g_disclaim_total = len(g_with_disclaim)

        genre_breakdown[genre] = {
            'n_total': len(g_df),
            'n_with_disclaimer': g_disclaim_total,
            'n_warn_and_answer': g_disclaim_unsafe,
            'warn_and_answer_rate': float(g_disclaim_unsafe / g_disclaim_total) if g_disclaim_total > 0 else 0.0
        }

    results['by_genre'] = genre_breakdown

    # Statistical test: Is disclaimer presence associated with harm classification?
    # Fisher's exact on the 2x2 table
    table = np.array([
        [disclaim_safe, disclaim_unsafe],
        [no_disclaim_safe, no_disclaim_unsafe]
    ])

    if table.sum() > 0 and table.min() >= 0:
        fisher_or, fisher_p = stats.fisher_exact(table)
        results['fisher_test'] = {
            'odds_ratio': float(fisher_or),
            'p_value': float(fisher_p),
            'interpretation': 'Tests whether disclaimer presence is independent of harm classification'
        }
    else:
        results['fisher_test'] = {'error': 'Invalid contingency table'}

    return results


def abliterated_vs_control_analysis(df):
    """Compare abliterated models vs control models."""
    abl_df = df[df['is_abliterated'] == True]
    ctrl_df = df[df['is_control'] == True]

    results = {
        'abliterated': {
            'n': len(abl_df),
            'rate': abl_df['disclaimer_int'].mean(),
            'by_genre': abl_df.groupby('genre')['disclaimer_int'].mean().to_dict()
        },
        'control': {
            'n': len(ctrl_df),
            'rate': ctrl_df['disclaimer_int'].mean(),
            'by_genre': ctrl_df.groupby('genre')['disclaimer_int'].mean().to_dict()
        }
    }

    # Statistical comparison
    if len(abl_df) > 0 and len(ctrl_df) > 0:
        contingency = pd.crosstab(df['is_abliterated'], df['has_disclaimer'])
        chi2, p_value, dof, _ = stats.chi2_contingency(contingency)
        results['chi2'] = chi2
        results['p_value'] = p_value

    return results


# ============================================================================
# Output Formatting
# ============================================================================

def generate_latex_tables(df, rates, violence_gap, per_model_gap, pairwise):
    """Generate LaTeX-formatted tables for the paper."""
    latex = []

    # Table 1: Disclaimer Rates by Genre
    latex.append(r"""
% Table 1: Disclaimer Rates by Genre (All Models)
\begin{table}[htbp]
\centering
\caption{Disclaimer Rates by Content Genre}
\label{tab:disclaimer-rates-genre}
\begin{tabular}{lrrr}
\toprule
\textbf{Genre} & \textbf{N} & \textbf{Disclaimers} & \textbf{Rate} \\
\midrule""")

    for genre, row in rates['by_genre'].iterrows():
        latex.append(f"{genre} & {int(row['n'])} & {int(row['disclaimers'])} & {row['rate']:.3f} \\\\")

    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Table 2: Disclaimer Rates by Model
    latex.append(r"""
% Table 2: Disclaimer Rates by Model
\begin{table}[htbp]
\centering
\caption{Disclaimer Rates by Model}
\label{tab:disclaimer-rates-model}
\begin{tabular}{lrrrl}
\toprule
\textbf{Model} & \textbf{N} & \textbf{Disclaimers} & \textbf{Rate} & \textbf{Type} \\
\midrule""")

    for model, row in rates['by_model'].iterrows():
        model_type = "Abliterated" if row['is_abliterated'] else "Control"
        latex.append(f"{model} & {int(row['n'])} & {int(row['disclaimers'])} & {row['rate']:.3f} & {model_type} \\\\")

    latex.append(r"""\bottomrule
\end{tabular}
\end{table}
""")

    # Table 3: Violence Gap Analysis
    latex.append(r"""
% Table 3: Violence Gap Analysis
\begin{table}[htbp]
\centering
\caption{Violence Gap: Violence vs.\ Other Genres}
\label{tab:violence-gap}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Category} & \textbf{N} & \textbf{Rate} & \textbf{OR} & \textbf{95\% CI} & \textbf{$p$-value} \\
\midrule""")

    ci = violence_gap['odds_ratio_ci']
    ci_str = f"[{ci[0]:.2f}, {ci[1]:.2f}]" if not np.isnan(ci[0]) else "---"
    latex.append(f"Violence & {violence_gap['violence_n']} & {violence_gap['violence_rate']:.3f} & \\multirow{{2}}{{*}}{{{violence_gap['odds_ratio']:.2f}}} & \\multirow{{2}}{{*}}{{{ci_str}}} & \\multirow{{2}}{{*}}{{${violence_gap['p_value']:.4f}$}} \\\\")
    latex.append(f"Other & {violence_gap['other_n']} & {violence_gap['other_rate']:.3f} & & & \\\\")

    latex.append(r"""\midrule
\multicolumn{6}{l}{\textit{Effect size: Cram\'er's V = """ + f"{violence_gap['cramers_v']:.3f}" + r"""}} \\
\bottomrule
\end{tabular}
\end{table}
""")

    # Table 4: Per-Model Violence Gap
    latex.append(r"""
% Table 4: Violence Gap by Model
\begin{table}[htbp]
\centering
\caption{Violence Gap by Model}
\label{tab:violence-gap-model}
\begin{tabular}{lrrrrl}
\toprule
\textbf{Model} & \textbf{Violence} & \textbf{Other} & \textbf{Gap} & \textbf{$p$-value} & \textbf{Sig.} \\
\midrule""")

    for model, data in sorted(per_model_gap.items()):
        sig = "*" if data['fisher_p'] < 0.05 else ""
        sig = "**" if data['fisher_p'] < 0.01 else sig
        sig = "***" if data['fisher_p'] < 0.001 else sig
        p_str = f"{data['fisher_p']:.4f}" if not np.isnan(data['fisher_p']) else "---"
        latex.append(f"{model} & {data['violence_rate']:.3f} & {data['other_rate']:.3f} & {data['gap']:.3f} & {p_str} & {sig} \\\\")

    latex.append(r"""\bottomrule
\multicolumn{6}{l}{\footnotesize $^{*}p<0.05$, $^{**}p<0.01$, $^{***}p<0.001$ (Fisher's exact test)} \\
\end{tabular}
\end{table}
""")

    # Table 5: Pairwise Genre Comparisons with Holm correction
    latex.append(r"""
% Table 5: Pairwise Genre Comparisons (Fisher's Exact with Holm Correction)
\begin{table}[htbp]
\centering
\caption{Pairwise Genre Comparisons (Holm-adjusted)}
\label{tab:pairwise-comparisons}
\begin{tabular}{lrrrrr}
\toprule
\textbf{Comparison} & \textbf{Rate 1} & \textbf{Rate 2} & \textbf{OR} & \textbf{$p$} & \textbf{$p_{\text{Holm}}$} \\
\midrule""")

    for comp in pairwise:
        p_holm = comp.get('p_value_holm', comp['p_value'])
        sig = "*" if p_holm < 0.05 else ""
        sig = "**" if p_holm < 0.01 else sig
        sig = "***" if p_holm < 0.001 else sig
        latex.append(f"{comp['comparison']} & {comp['genre1_disclaimer_rate']:.3f} & {comp['genre2_disclaimer_rate']:.3f} & {comp['odds_ratio']:.2f} & {comp['p_value']:.4f} & {p_holm:.4f}{sig} \\\\")

    latex.append(r"""\bottomrule
\multicolumn{6}{l}{\footnotesize $^{*}p_{\text{Holm}}<0.05$, $^{**}p_{\text{Holm}}<0.01$, $^{***}p_{\text{Holm}}<0.001$} \\
\end{tabular}
\end{table}
""")

    return '\n'.join(latex)


def create_json_output(df, rates, violence_gap, per_model_gap, pairwise, chi_sq, abl_ctrl, mixed_effects=None, warn_answer=None):
    """Create JSON output with all statistics."""

    # Convert pandas objects to serializable format
    def convert_to_serializable(obj):
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, tuple):
            return list(obj)
        return obj

    output = {
        'summary': {
            'total_records': len(df),
            'models': df['model_clean'].nunique(),
            'genres': df['genre'].unique().tolist(),
            'overall_disclaimer_rate': float(df['disclaimer_int'].mean())
        },
        'rates_by_genre': rates['by_genre'].to_dict(),
        'rates_by_model': rates['by_model'].drop(columns=['is_abliterated']).to_dict(),
        'chi_square_test': {
            'chi2': float(chi_sq['chi2']),
            'p_value': float(chi_sq['p_value']),
            'dof': int(chi_sq['dof']),
            'cramers_v': float(chi_sq['cramers_v'])
        },
        'violence_gap_hypothesis': {
            'violence_rate': float(violence_gap['violence_rate']),
            'other_rate': float(violence_gap['other_rate']),
            'gap': float(violence_gap['rate_difference']),
            'chi2': float(violence_gap['chi2']),
            'p_value': float(violence_gap['p_value']),
            'cramers_v': float(violence_gap['cramers_v']),
            'odds_ratio': float(violence_gap['odds_ratio']),
            'odds_ratio_ci': [float(x) if not np.isnan(x) else None for x in violence_gap['odds_ratio_ci']],
            'fisher_exact_p': float(violence_gap['fisher_p'])
        },
        'per_model_violence_gap': {
            model: {k: convert_to_serializable(v) for k, v in data.items()}
            for model, data in per_model_gap.items()
        },
        'pairwise_comparisons': [
            {
                'comparison': comp['comparison'],
                'odds_ratio': float(comp['odds_ratio']),
                'p_value': float(comp['p_value']),
                'p_value_holm': float(comp.get('p_value_holm', comp['p_value'])),
                'significant_holm': bool(comp.get('significant_holm', comp['p_value'] < 0.05)),
                'rate_1': float(comp['genre1_disclaimer_rate']),
                'rate_2': float(comp['genre2_disclaimer_rate'])
            }
            for comp in pairwise
        ],
        'abliterated_vs_control': {
            k: convert_to_serializable(v) for k, v in abl_ctrl.items()
        }
    }

    # Add mixed-effects results if available
    if mixed_effects is not None:
        output['mixed_effects_regression'] = mixed_effects

    # Add warn-and-answer analysis if available
    if warn_answer is not None and 'error' not in warn_answer:
        output['warn_and_answer_analysis'] = warn_answer

    return output


# ============================================================================
# Main Execution
# ============================================================================

def main():
    print("=" * 70)
    print("Genre-Mimicry Statistical Analysis")
    print("=" * 70)

    # Setup paths
    data_dir = Path(__file__).parent.parent
    output_dir = Path(__file__).parent

    print(f"\nData directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    # Load data
    print("\n[1] Loading data...")
    df = load_all_data(data_dir)
    print(f"\nTotal records: {len(df)}")
    print(f"Models: {df['model_clean'].nunique()}")
    print(f"Genres: {df['genre'].unique().tolist()}")

    # Compute rates
    print("\n[2] Computing disclaimer rates...")
    rates = analyze_overall_rates(df)

    print("\n--- Rates by Genre ---")
    print(rates['by_genre'])

    print("\n--- Rates by Model ---")
    print(rates['by_model'])

    # Chi-square test
    print("\n[3] Running Chi-Square test (Genre vs Disclaimer)...")
    chi_sq = run_chi_square_test(df)
    print(f"  Chi-square: {chi_sq['chi2']:.4f}")
    print(f"  p-value: {chi_sq['p_value']:.6f}")
    print(f"  Cramer's V: {chi_sq['cramers_v']:.4f}")

    # Violence Gap test
    print("\n[4] Testing Violence Gap hypothesis...")
    violence_gap = test_violence_gap(df)
    print(f"  Violence disclaimer rate: {violence_gap['violence_rate']:.3f} (n={violence_gap['violence_n']})")
    print(f"  Other disclaimer rate: {violence_gap['other_rate']:.3f} (n={violence_gap['other_n']})")
    print(f"  Gap: {violence_gap['rate_difference']:.3f}")
    print(f"  Chi-square p-value: {violence_gap['p_value']:.6f}")
    print(f"  Fisher's exact p-value: {violence_gap['fisher_p']:.6f}")
    print(f"  Odds Ratio: {violence_gap['odds_ratio']:.3f}")
    print(f"  OR 95% CI: [{violence_gap['odds_ratio_ci'][0]:.3f}, {violence_gap['odds_ratio_ci'][1]:.3f}]")

    # Per-model Violence Gap
    print("\n[5] Violence Gap by model...")
    per_model_gap = test_violence_gap_per_model(df)
    for model, data in per_model_gap.items():
        sig = "***" if data['fisher_p'] < 0.001 else "**" if data['fisher_p'] < 0.01 else "*" if data['fisher_p'] < 0.05 else ""
        print(f"  {model}: Violence={data['violence_rate']:.3f}, Other={data['other_rate']:.3f}, Gap={data['gap']:.3f} {sig}")

    # Pairwise comparisons
    print("\n[6] Pairwise genre comparisons...")
    pairwise = pairwise_genre_comparisons(df)
    for comp in pairwise:
        print(f"  {comp['comparison']}: OR={comp['odds_ratio']:.3f}, p={comp['p_value']:.4f}")

    # Abliterated vs Control
    print("\n[7] Abliterated vs Control analysis...")
    abl_ctrl = abliterated_vs_control_analysis(df)
    print(f"  Abliterated: rate={abl_ctrl['abliterated']['rate']:.3f} (n={abl_ctrl['abliterated']['n']})")
    print(f"  Control: rate={abl_ctrl['control']['rate']:.3f} (n={abl_ctrl['control']['n']})")

    # Mixed-effects logistic regression
    print("\n[8] Running mixed-effects regression (disclaimer ~ genre + (1|model))...")
    mixed_effects = run_mixed_effects_logistic(df)

    # Warn-and-answer analysis (disclaimer × Llama Guard harm)
    print("\n[9] Analyzing warn-and-answer patterns (disclaimer × harm classification)...")
    harm_scores_path = output_dir / 'harm_scores_ollama.jsonl'
    warn_answer = analyze_warn_and_answer(harm_scores_path)
    if 'error' not in warn_answer:
        print(f"  Total responses: {warn_answer['n_total']}")
        print(f"  With disclaimer: {warn_answer['n_with_disclaimer']}")
        print(f"  Guard-unsafe: {warn_answer['n_guard_unsafe']}")
        print(f"\n  Warn-and-answer pattern (disclaimer + unsafe):")
        print(f"    Count: {warn_answer['warn_and_answer']['count']}")
        print(f"    Rate given disclaimer: {warn_answer['warn_and_answer']['rate_given_disclaimer']:.1%}")
        print(f"\n  By genre:")
        for genre, data in warn_answer['by_genre'].items():
            if data['n_with_disclaimer'] > 0:
                print(f"    {genre}: {data['warn_and_answer_rate']:.1%} warn-and-answer ({data['n_warn_and_answer']}/{data['n_with_disclaimer']})")
    else:
        print(f"  [WARNING] {warn_answer['error']}")
        warn_answer = None

    # Generate outputs
    print("\n[10] Generating output files...")

    # JSON output
    json_output = create_json_output(df, rates, violence_gap, per_model_gap, pairwise, chi_sq, abl_ctrl, mixed_effects, warn_answer)
    json_path = output_dir / 'analysis_results.json'
    with open(json_path, 'w') as f:
        json.dump(json_output, f, indent=2)
    print(f"  Saved: {json_path}")

    # LaTeX tables
    latex_output = generate_latex_tables(df, rates, violence_gap, per_model_gap, pairwise)
    latex_path = output_dir / 'results_tables.tex'
    with open(latex_path, 'w') as f:
        f.write(latex_output)
    print(f"  Saved: {latex_path}")

    # Summary CSV
    summary_df = df.groupby(['model_clean', 'genre']).agg(
        n=('has_disclaimer', 'count'),
        disclaimers=('disclaimer_int', 'sum'),
        rate=('disclaimer_int', 'mean')
    ).round(4)
    csv_path = output_dir / 'summary_by_model_genre.csv'
    summary_df.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("=" * 70)

    return df, json_output


if __name__ == '__main__':
    df, results = main()
