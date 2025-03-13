"""
Analysis of operational impacts and safety metrics for wildlife strikes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class SafetyMetric:
    name: str
    value: float
    trend: float
    risk_level: str  # High, Medium, Low
    description: str

def analyze_operational_delays(
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Analyze operational delays caused by wildlife strikes
    """
    # Calculate delay metrics
    delay_metrics = {
        'total_impact_hours': df['OUT_OF_SERVICE_HRS'].sum(),
        'average_out_of_service_hours': df['OUT_OF_SERVICE_HRS'].mean(),
        'median_out_of_service_hours': df['OUT_OF_SERVICE_HRS'].median(),
        'max_delay': df['OUT_OF_SERVICE_HRS'].max(),
        'total_affected_flights': len(df[df['OUT_OF_SERVICE_HRS'] > 0])
    }
    
    # Calculate delay distribution percentiles
    percentiles = np.percentile(
        df['OUT_OF_SERVICE_HRS'].dropna(),
        [25, 50, 75, 90, 95]
    )
    
    delay_metrics.update({
        '25th_percentile': percentiles[0],
        'median': percentiles[1],
        '75th_percentile': percentiles[2],
        '90th_percentile': percentiles[3],
        '95th_percentile': percentiles[4]
    })
    
    return delay_metrics

def analyze_phase_of_flight_risk(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Analyze risk levels for different phases of flight
    """
    # Calculate risk metrics by phase
    phase_risk = df.groupby('PHASE_OF_FLT').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['mean', 'sum'],
        'HAS_DAMAGE': 'mean',
        'OUT_OF_SERVICE_HRS': 'mean'
    })
    
    # Flatten column names
    phase_risk.columns = ['_'.join(col).strip() for col in phase_risk.columns.values]
    
    # Calculate incident rate
    total_incidents = len(df)
    phase_risk['incident_rate'] = phase_risk['INDEX NR_count'] / total_incidents
    
    # Calculate risk score
    phase_risk['risk_score'] = (
        phase_risk['DAMAGE_SCORE_mean'] *
        phase_risk['incident_rate'] *
        np.log1p(phase_risk['TOTAL_COST_mean'])
    )
    
    return phase_risk

def analyze_aircraft_vulnerability(
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Analyze aircraft vulnerability patterns
    """
    # Calculate vulnerability metrics by aircraft type
    vulnerability = df.groupby(['AC_MASS', 'AC_CLASS']).agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['mean', 'sum'],
        'HAS_DAMAGE': 'mean'
    })
    
    # Flatten column names
    vulnerability.columns = ['_'.join(col).strip() for col in vulnerability.columns.values]
    
    # Identify critical components
    damage_cols = [col for col in df.columns if col.startswith('DAM_')]
    damage_rates = df[damage_cols].mean()
    
    critical_components = {
        'high_damage_rate': damage_rates[damage_rates > damage_rates.mean() + damage_rates.std()].index.tolist(),
        'high_repair_cost': df.groupby('COMPONENT')['COST_REPAIRS'].mean().nlargest(5).index.tolist()
    }
    
    return vulnerability, critical_components

def calculate_safety_metrics(
    df: pd.DataFrame
) -> Dict[str, float]:
    """
    Calculate comprehensive safety metrics
    """
    total_incidents = len(df)
    incidents_with_damage = len(df[df['HAS_DAMAGE'] == 1])
    incidents_with_injury = len(df[df['NR_INJURIES'] > 0])
    
    safety_metrics = {
        'incident_rate': total_incidents / df['YEAR'].nunique(),  # Annual rate
        'damage_rate': incidents_with_damage / total_incidents,
        'injury_rate': incidents_with_injury / total_incidents,
        'average_damage_score': df['DAMAGE_SCORE'].mean(),
        'severe_incident_rate': len(df[df['DAMAGE_SCORE'] > 3]) / total_incidents
    }
    
    return safety_metrics

def analyze_warning_effectiveness(
    df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Analyze effectiveness of warning systems and prevention measures
    """
    # Calculate metrics for incidents with and without warnings
    warning_metrics = {}
    
    for warning_type in ['WARNING_ISSUED', 'PREVENTION_METHOD']:
        if warning_type in df.columns:
            with_warning = df[df[warning_type].notna()]
            without_warning = df[df[warning_type].isna()]
            
            warning_metrics[warning_type] = {
                'damage_rate_with': with_warning['HAS_DAMAGE'].mean(),
                'damage_rate_without': without_warning['HAS_DAMAGE'].mean(),
                'avg_cost_with': with_warning['TOTAL_COST'].mean(),
                'avg_cost_without': without_warning['TOTAL_COST'].mean(),
                'effectiveness': 1 - (
                    with_warning['DAMAGE_SCORE'].mean() /
                    without_warning['DAMAGE_SCORE'].mean()
                )
            }
    
    return warning_metrics

def calculate_risk_scores(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate comprehensive risk scores for different factors
    """
    risk_factors = pd.DataFrame()
    
    # Calculate base incident rates
    for factor in ['AIRPORT', 'OPERATOR', 'AC_CLASS']:
        group_metrics = df.groupby(factor).agg({
            'INDEX NR': 'count',
            'DAMAGE_SCORE': 'mean',
            'TOTAL_COST': ['mean', 'sum'],
            'HAS_DAMAGE': 'mean'
        })
        
        # Flatten column names
        group_metrics.columns = ['_'.join(col).strip() for col in group_metrics.columns.values]
        
        # Calculate risk score
        total_incidents = len(df)
        group_metrics['incident_rate'] = group_metrics['INDEX NR_count'] / total_incidents
        
        group_metrics['risk_score'] = (
            group_metrics['DAMAGE_SCORE_mean'] *
            group_metrics['incident_rate'] *
            np.log1p(group_metrics['TOTAL_COST_mean'])
        )
        
        risk_factors[factor] = group_metrics['risk_score']
    
    return risk_factors

def analyze_operational_conditions(
    df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Analyze impact of operational conditions on strike risk
    """
    condition_analysis = {}
    
    # Analyze weather conditions
    if 'SKY' in df.columns:
        weather_impact = df.groupby('SKY').agg({
            'INDEX NR': 'count',
            'DAMAGE_SCORE': ['mean', 'std'],
            'TOTAL_COST': 'mean',
            'HAS_DAMAGE': 'mean'
        })
        condition_analysis['weather'] = weather_impact
    
    # Analyze visibility conditions
    if 'VISIBILITY' in df.columns:
        visibility_impact = df.groupby('VISIBILITY').agg({
            'INDEX NR': 'count',
            'DAMAGE_SCORE': ['mean', 'std'],
            'TOTAL_COST': 'mean',
            'HAS_DAMAGE': 'mean'
        })
        condition_analysis['visibility'] = visibility_impact
    
    # Analyze time of day
    time_impact = df.groupby('TIME_OF_DAY').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': 'mean',
        'HAS_DAMAGE': 'mean'
    })
    condition_analysis['time_of_day'] = time_impact
    
    return condition_analysis

def calculate_severity_index(
    df: pd.DataFrame
) -> pd.Series:
    """
    Calculate a composite severity index for incidents
    """
    # Create standardizer
    scaler = StandardScaler()
    
    # Select severity components
    severity_components = [
        'DAMAGE_SCORE',
        'TOTAL_COST',
        'OUT_OF_SERVICE_HRS',
        'NR_INJURIES'
    ]
    
    # Standardize components
    standardized = pd.DataFrame(
        scaler.fit_transform(df[severity_components]),
        columns=severity_components,
        index=df.index
    )
    
    # Calculate weighted severity index
    weights = {
        'DAMAGE_SCORE': 0.4,
        'TOTAL_COST': 0.3,
        'OUT_OF_SERVICE_HRS': 0.2,
        'NR_INJURIES': 0.1
    }
    
    severity_index = sum(
        standardized[component] * weight
        for component, weight in weights.items()
    )
    
    return severity_index

def analyze_mitigation_effectiveness(
    df: pd.DataFrame
) -> Dict[str, Dict[str, float]]:
    """
    Analyze effectiveness of various mitigation strategies
    """
    mitigation_analysis = {}
    
    # Analyze each mitigation method if available
    mitigation_methods = [col for col in df.columns if col.startswith('MITIGATION_')]
    
    for method in mitigation_methods:
        with_mitigation = df[df[method] == 1]
        without_mitigation = df[df[method] == 0]
        
        effectiveness_metrics = {
            'incident_rate_reduction': 1 - (
                len(with_mitigation) / len(without_mitigation)
            ),
            'damage_rate_reduction': 1 - (
                with_mitigation['HAS_DAMAGE'].mean() /
                without_mitigation['HAS_DAMAGE'].mean()
            ),
            'cost_reduction': 1 - (
                with_mitigation['TOTAL_COST'].mean() /
                without_mitigation['TOTAL_COST'].mean()
            ),
            'severity_reduction': 1 - (
                with_mitigation['DAMAGE_SCORE'].mean() /
                without_mitigation['DAMAGE_SCORE'].mean()
            )
        }
        
        mitigation_analysis[method] = effectiveness_metrics
    
    return mitigation_analysis