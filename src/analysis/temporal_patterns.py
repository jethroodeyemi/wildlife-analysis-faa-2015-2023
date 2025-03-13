"""
Advanced temporal pattern analysis for wildlife strikes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import statsmodels.api as sm

def analyze_long_term_trends(
    df: pd.DataFrame,
    freq: str = 'M'
) -> Dict[str, Dict[str, float]]:
    """
    Analyze long-term trends in various metrics
    """
    metrics = {
        'incident_count': lambda x: len(x),
        'damage_rate': lambda x: x['HAS_DAMAGE'].mean(),
        'avg_cost': lambda x: x['TOTAL_COST'].mean(),
        'severe_damage_rate': lambda x: (x['DAMAGE'] == 'S').mean()
    }
    
    results = {}
    
    for metric_name, metric_func in metrics.items():
        # Create time series
        ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq=freq)).apply(metric_func)
        
        # Perform trend analysis
        trend_test = sm.OLS(
            ts.values,
            sm.add_constant(np.arange(len(ts)))
        ).fit()
        
        # Perform stationarity test
        adf_test = adfuller(ts.dropna())
        
        results[metric_name] = {
            'trend_coefficient': trend_test.params[1],
            'trend_p_value': trend_test.pvalues[1],
            'is_significant': trend_test.pvalues[1] < 0.05,
            'r_squared': trend_test.rsquared,
            'is_stationary': adf_test[1] < 0.05
        }
    
    return results

def analyze_cyclical_patterns(
    df: pd.DataFrame,
    freq: str = 'M'
) -> Dict[str, pd.Series]:
    """
    Analyze cyclical patterns in the data
    """
    # Create time series of incidents
    ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq=freq)).size()
    
    # Perform seasonal decomposition
    decomposition = seasonal_decompose(ts, period=12)
    
    # Calculate seasonal indices
    seasonal_idx = pd.Series(
        decomposition.seasonal[:12],
        index=range(1, 13)
    )
    
    # Calculate trend strength
    trend_strength = 1 - (np.var(decomposition.resid) / np.var(ts - decomposition.seasonal))
    
    # Calculate seasonal strength
    seasonal_strength = 1 - (np.var(decomposition.resid) / 
                           np.var(ts - decomposition.trend))
    
    return {
        'seasonal_indices': seasonal_idx,
        'trend': decomposition.trend,
        'seasonal': decomposition.seasonal,
        'residual': decomposition.resid,
        'trend_strength': trend_strength,
        'seasonal_strength': seasonal_strength
    }

def analyze_monthly_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze monthly patterns in various metrics
    """
    monthly_metrics = df.groupby('MONTH').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['mean', 'sum'],
        'HEIGHT': 'mean',
        'SPEED': 'mean',
        'HAS_DAMAGE': 'mean'
    })
    
    # Flatten column names
    monthly_metrics.columns = ['_'.join(col).strip() for col in monthly_metrics.columns.values]
    
    # Calculate month-to-month changes
    monthly_changes = monthly_metrics.pct_change()
    
    # Identify peak months
    peak_months = {
        col: monthly_metrics[col].idxmax()
        for col in monthly_metrics.columns
    }
    
    return {
        'monthly_metrics': monthly_metrics,
        'monthly_changes': monthly_changes,
        'peak_months': peak_months
    }

def analyze_time_of_day_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns in different times of day
    """
    # Create time bins
    df['HOUR'] = pd.to_datetime(df['TIME']).dt.hour
    df['TIME_BIN'] = pd.cut(
        df['HOUR'],
        bins=[0, 6, 12, 18, 24],
        labels=['Night', 'Morning', 'Afternoon', 'Evening']
    )
    
    # Analyze patterns by time of day
    time_metrics = df.groupby('TIME_BIN').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['mean', 'sum'],
        'HEIGHT': 'mean',
        'SPEED': 'mean',
        'HAS_DAMAGE': 'mean'
    })
    
    # Flatten column names
    time_metrics.columns = ['_'.join(col).strip() for col in time_metrics.columns.values]
    
    # Calculate relative risk by time of day
    overall_damage_rate = df['HAS_DAMAGE'].mean()
    time_metrics['relative_risk'] = (
        time_metrics['HAS_DAMAGE_mean'] / overall_damage_rate
    )
    
    return {
        'time_metrics': time_metrics,
        'peak_time': time_metrics['INDEX NR_count'].idxmax(),
        'highest_risk_time': time_metrics['relative_risk'].idxmax()
    }

def analyze_multi_year_patterns(
    df: pd.DataFrame,
    min_years: int = 5
) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns across multiple years
    """
    # Create year and month columns
    df['YEAR'] = pd.to_datetime(df['INCIDENT_DATE']).dt.year
    df['MONTH'] = pd.to_datetime(df['INCIDENT_DATE']).dt.month
    
    # Create pivot table for month-year analysis
    monthly_counts = pd.pivot_table(
        df,
        values='INDEX NR',
        index='MONTH',
        columns='YEAR',
        aggfunc='count',
        fill_value=0
    )
    
    # Calculate year-over-year changes
    yoy_changes = df.groupby('YEAR').agg({
        'INDEX NR': 'count',
        'TOTAL_COST': 'sum',
        'DAMAGE_SCORE': 'mean'
    }).pct_change()
    
    # Identify long-term patterns
    long_term_patterns = pd.DataFrame()
    
    # Calculate average monthly pattern for each metric
    for metric in ['INDEX NR', 'DAMAGE_SCORE', 'TOTAL_COST']:
        monthly_avg = df.groupby('MONTH')[metric].mean()
        monthly_std = df.groupby('MONTH')[metric].std()
        
        long_term_patterns[f'{metric}_avg'] = monthly_avg
        long_term_patterns[f'{metric}_std'] = monthly_std
    
    return {
        'monthly_counts': monthly_counts,
        'yoy_changes': yoy_changes,
        'long_term_patterns': long_term_patterns
    }