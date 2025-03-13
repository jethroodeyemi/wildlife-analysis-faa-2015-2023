"""
Advanced statistical analysis for wildlife strikes
"""
import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.seasonal import seasonal_decompose
from typing import Dict, List, Tuple, Optional

def perform_hypothesis_tests(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """
    Perform various hypothesis tests on the data
    """
    test_results = {}
    
    # Test 1: Are strikes more frequent during certain times of day?
    time_counts = df['TIME_OF_DAY'].value_counts()
    expected_freq = len(df) / len(time_counts)
    chi2, p_value = stats.chisquare(time_counts, [expected_freq] * len(time_counts))
    test_results['time_of_day_distribution'] = {
        'statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Test 2: Is there a relationship between damage severity and aircraft size?
    damage_by_size = pd.crosstab(df['AC_MASS'], df['DAMAGE'])
    chi2, p_value = stats.chi2_contingency(damage_by_size)[:2]
    test_results['damage_size_relationship'] = {
        'statistic': chi2,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Test 3: Compare heights across different species
    species_heights = {species: group['HEIGHT'].dropna() 
                      for species, group in df.groupby('SPECIES')}
    f_stat, p_value = stats.f_oneway(*[heights for heights in species_heights.values()])
    test_results['height_distribution'] = {
        'statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    return test_results

def analyze_cost_distribution(df: pd.DataFrame) -> Dict[str, float]:
    """
    Analyze the statistical distribution of costs
    """
    costs = df['TOTAL_COST'].dropna()
    
    # Test for normality
    normality_stat, normality_p = stats.normaltest(costs)
    
    # Calculate distribution parameters
    distribution_params = {
        'mean': costs.mean(),
        'median': costs.median(),
        'std': costs.std(),
        'skewness': stats.skew(costs),
        'kurtosis': stats.kurtosis(costs),
        'normality_test_stat': normality_stat,
        'normality_p_value': normality_p,
        'is_normal': normality_p > 0.05
    }
    
    return distribution_params

def perform_trend_analysis(
    df: pd.DataFrame,
    metric: str,
    frequency: str = 'M'
) -> Dict[str, float]:
    """
    Perform trend analysis on time series data
    """
    # Create time series
    ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq=frequency))[metric].mean()
    
    # Perform Mann-Kendall trend test
    trend, h, p, z = stats.kendalltau(ts.index.astype(int), ts.values)
    
    # Calculate seasonal decomposition
    decomp = seasonal_decompose(ts, period=12)
    
    return {
        'trend_coefficient': trend,
        'p_value': p,
        'significant': p < 0.05,
        'seasonality_strength': 1 - (np.var(decomp.resid) / np.var(ts - decomp.trend))
    }

def compare_groups(
    df: pd.DataFrame,
    group_col: str,
    metric_col: str
) -> Dict[str, Dict[str, float]]:
    """
    Perform statistical comparison between groups
    """
    results = {}
    
    # Perform one-way ANOVA
    groups = [group[metric_col].dropna() for name, group in df.groupby(group_col)]
    f_stat, p_value = stats.f_oneway(*groups)
    
    results['anova'] = {
        'statistic': f_stat,
        'p_value': p_value,
        'significant': p_value < 0.05
    }
    
    # Perform Tukey's HSD test for pairwise comparisons
    tukey = pairwise_tukeyhsd(
        df[metric_col],
        df[group_col]
    )
    
    # Convert Tukey results to dictionary
    tukey_results = []
    for row in tukey._results_table.data[1:]:
        tukey_results.append({
            'group1': row[0],
            'group2': row[1],
            'mean_diff': row[2],
            'p_value': row[3],
            'significant': row[3] < 0.05
        })
    
    results['tukey_hsd'] = tukey_results
    
    return results

def calculate_confidence_intervals(
    df: pd.DataFrame,
    metric: str,
    confidence_level: float = 0.95
) -> Dict[str, Tuple[float, float]]:
    """
    Calculate confidence intervals for various metrics
    """
    intervals = {}
    
    # Overall confidence interval
    data = df[metric].dropna()
    mean = data.mean()
    sem = stats.sem(data)
    ci = stats.t.interval(confidence_level, len(data)-1, mean, sem)
    intervals['overall'] = (ci[0], ci[1])
    
    # Confidence intervals by damage category
    for damage_type in df['DAMAGE'].unique():
        damage_data = df[df['DAMAGE'] == damage_type][metric].dropna()
        if len(damage_data) > 1:  # Need at least 2 points for CI
            mean = damage_data.mean()
            sem = stats.sem(damage_data)
            ci = stats.t.interval(confidence_level, len(damage_data)-1, mean, sem)
            intervals[f'damage_{damage_type}'] = (ci[0], ci[1])
    
    return intervals

def analyze_correlations(
    df: pd.DataFrame,
    variables: List[str],
    method: str = 'spearman'
) -> Dict[str, pd.DataFrame]:
    """
    Analyze correlations between variables
    """
    # Calculate correlation matrix
    corr_matrix = df[variables].corr(method=method)
    
    # Calculate p-values
    p_values = pd.DataFrame(np.zeros_like(corr_matrix), 
                          index=corr_matrix.index,
                          columns=corr_matrix.columns)
    
    for i in range(len(variables)):
        for j in range(len(variables)):
            if i != j:
                if method == 'spearman':
                    coef, p = stats.spearmanr(df[variables[i]], df[variables[j]])
                else:
                    coef, p = stats.pearsonr(df[variables[i]], df[variables[j]])
                p_values.iloc[i,j] = p
    
    return {
        'correlations': corr_matrix,
        'p_values': p_values,
        'significant_pairs': (p_values < 0.05) & (corr_matrix.abs() > 0.3)
    }

def analyze_seasonal_patterns(
    df: pd.DataFrame,
    column: str,
    freq: str = 'M'
) -> Dict[str, pd.Series]:
    """
    Analyze seasonal patterns in the data
    """
    # Create time series
    ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq=freq))[column].count()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts, period=12)
    
    return {
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid
    }

def calculate_risk_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate risk metrics for different categories
    """
    risk_metrics = df.groupby(['OPERATOR_CATEGORY', 'AC_CLASS']).agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['sum', 'mean'],
        'HAS_DAMAGE': 'mean'
    })
    
    # Flatten column names
    risk_metrics.columns = ['_'.join(col).strip() for col in risk_metrics.columns.values]
    
    # Calculate risk score
    risk_metrics['risk_score'] = (
        risk_metrics['DAMAGE_SCORE_mean'] * 
        risk_metrics['HAS_DAMAGE_mean'] * 
        np.log1p(risk_metrics['TOTAL_COST_mean'])
    )
    
    return risk_metrics

def perform_species_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze wildlife species impact and patterns
    """
    species_analysis = df.groupby('SPECIES').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['sum', 'mean'],
        'HEIGHT': ['mean', 'median'],
        'SPEED': ['mean', 'median'],
        'HAS_DAMAGE': 'mean'
    })
    
    # Flatten column names
    species_analysis.columns = ['_'.join(col).strip() for col in species_analysis.columns.values]
    
    # Calculate strike rate per 10,000 incidents
    total_incidents = len(df)
    species_analysis['strike_rate'] = (species_analysis['INDEX NR_count'] / total_incidents) * 10000
    
    return species_analysis

def analyze_geographic_patterns(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Analyze geographic patterns and regional risk factors
    """
    # Regional analysis
    regional_analysis = df.groupby(['STATE', 'FAAREGION']).agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': 'mean',
        'TOTAL_COST': ['sum', 'mean'],
        'HAS_DAMAGE': 'mean'
    })
    
    # Calculate spatial statistics
    spatial_stats = {}
    
    # Calculate geographic centroid of incidents
    spatial_stats['incident_centroid'] = {
        'latitude': df['AIRPORT_LATITUDE'].mean(),
        'longitude': df['AIRPORT_LONGITUDE'].mean()
    }
    
    # Calculate geographic dispersion
    spatial_stats['geographic_dispersion'] = {
        'latitude_std': df['AIRPORT_LATITUDE'].std(),
        'longitude_std': df['AIRPORT_LONGITUDE'].std()
    }
    
    return regional_analysis, spatial_stats

def perform_comparative_analysis(
    df: pd.DataFrame,
    group1_mask: pd.Series,
    group2_mask: pd.Series,
    metrics: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Perform comparative analysis between two groups
    """
    results = {}
    
    for metric in metrics:
        group1_data = df[group1_mask][metric]
        group2_data = df[group2_mask][metric]
        
        # Perform t-test
        t_stat, p_value = stats.ttest_ind(
            group1_data.dropna(),
            group2_data.dropna()
        )
        
        # Calculate effect size (Cohen's d)
        effect_size = (group1_data.mean() - group2_data.mean()) / np.sqrt(
            ((len(group1_data) - 1) * group1_data.std()**2 + 
             (len(group2_data) - 1) * group2_data.std()**2) /
            (len(group1_data) + len(group2_data) - 2)
        )
        
        results[metric] = {
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'significant': p_value < 0.05
        }
    
    return results

def calculate_temporal_risk_factors(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate risk factors based on temporal patterns
    """
    temporal_risk = pd.DataFrame()
    
    # Time of day analysis
    tod_risk = df.groupby('TIME_OF_DAY').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': 'mean',
        'HAS_DAMAGE': 'mean',
        'TOTAL_COST': 'mean'
    })
    
    # Seasonal analysis
    season_risk = df.groupby('SEASON').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': 'mean',
        'HAS_DAMAGE': 'mean',
        'TOTAL_COST': 'mean'
    })
    
    # Monthly analysis
    month_risk = df.groupby('MONTH').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': 'mean',
        'HAS_DAMAGE': 'mean',
        'TOTAL_COST': 'mean'
    })
    
    return {
        'time_of_day': tod_risk,
        'seasonal': season_risk,
        'monthly': month_risk
    }