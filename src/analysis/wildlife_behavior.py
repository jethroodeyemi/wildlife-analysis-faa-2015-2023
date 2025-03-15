"""
Analysis of wildlife species behavior and impact patterns
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def analyze_species_behavior(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze behavioral patterns of different wildlife species
    """
    # Define a safe mode function that handles empty results
    def safe_mode(x):
        mode_result = x.mode()
        if len(mode_result) > 0:
            return mode_result.iloc[0]
        return np.nan
    
    species_behavior = df.groupby('SPECIES').agg({
        'INDEX_NR': 'count',
        'HEIGHT': ['mean', 'std', 'median'],
        'SPEED': ['mean', 'std'],
        'TIME_OF_DAY': safe_mode,
        'INCIDENT_MONTH': safe_mode,
        'PRECIPITATION': safe_mode,
        'SKY': safe_mode
    })
    
    # Flatten column names
    species_behavior.columns = ['_'.join(col).strip() for col in species_behavior.columns.values]
    
    # Calculate frequency metrics
    total_incidents = len(df)
    species_behavior['frequency'] = species_behavior['INDEX_NR_count'] / total_incidents
    
    return species_behavior

def identify_species_clusters(
    df: pd.DataFrame,
    n_clusters: int = 5
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Identify clusters of species with similar characteristics
    """
    # Prepare features for clustering
    features = ['HEIGHT', 'SPEED', 'DAMAGE_SCORE', 'TOTAL_COST']
    species_metrics = df.groupby('SPECIES')[features].agg(['mean', 'std']).fillna(0)
    
    # Flatten column names
    species_metrics.columns = ['_'.join(col) for col in species_metrics.columns]
    
    # Scale features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(species_metrics)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Add cluster labels to metrics
    species_metrics['cluster'] = clusters
    
    # Analyze cluster characteristics
    cluster_profiles = {}
    for i in range(n_clusters):
        cluster_species = species_metrics[species_metrics['cluster'] == i].index.tolist()
        cluster_profiles[f'cluster_{i}'] = {
            'species': cluster_species,
            'size': len(cluster_species),
            'avg_height': species_metrics[species_metrics['cluster'] == i]['HEIGHT_mean'].mean(),
            'avg_damage': species_metrics[species_metrics['cluster'] == i]['DAMAGE_SCORE_mean'].mean()
        }
    
    return species_metrics, cluster_profiles

def analyze_species_impact(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze the impact of different species on aviation safety
    """
    # Calculate impact metrics
    impact_metrics = df.groupby('SPECIES').agg({
        'INDEX_NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['sum', 'mean'],
        'INDICATED_DAMAGE': 'mean',
        'NR_INJURIES': 'sum',
        'NR_FATALITIES': 'sum'
    })
    
    # Flatten column names
    impact_metrics.columns = ['_'.join(col).strip() for col in impact_metrics.columns.values]
    
    # Calculate risk score
    impact_metrics['risk_score'] = (
        impact_metrics['DAMAGE_SCORE_mean'] * 
        impact_metrics['INDICATED_DAMAGE_mean'] * 
        np.log1p(impact_metrics['TOTAL_COST_mean'])
    )
    
    # Check for non-zero risk scores to determine if categorization is possible
    non_zero_scores = impact_metrics['risk_score'].value_counts().shape[0]
    
    # Categorize species by risk level, handling duplicate bin edges
    if non_zero_scores > 4:
        try:
            risk_categories = pd.qcut(
                impact_metrics['risk_score'],
                q=4,
                labels=['Low', 'Medium', 'High', 'Very High'],
                duplicates='drop'  # Handle duplicate bin edges
            )
            impact_metrics['risk_category'] = risk_categories
        except ValueError:
            # If qcut still fails, use manual categorization based on percentiles
            thresholds = [
                impact_metrics['risk_score'].quantile(0),
                impact_metrics['risk_score'].quantile(0.25),
                impact_metrics['risk_score'].quantile(0.5),
                impact_metrics['risk_score'].quantile(0.75),
                impact_metrics['risk_score'].max() + 0.1  # Add small value to include max
            ]
            # Remove duplicates from thresholds
            thresholds = sorted(set(thresholds))
            
            # If we still don't have enough unique thresholds
            if len(thresholds) < 3:
                # Simple manual categorization
                conditions = [
                    impact_metrics['risk_score'] == 0,
                    impact_metrics['risk_score'] > 0
                ]
                choices = ['Low', 'High']
                impact_metrics['risk_category'] = np.select(conditions, choices, default='Medium')
            else:
                labels = ['Low', 'Medium', 'High', 'Very High'][:len(thresholds)-1]
                impact_metrics['risk_category'] = pd.cut(
                    impact_metrics['risk_score'], 
                    bins=thresholds, 
                    labels=labels,
                    include_lowest=True
                )
    else:
        # For very few unique values, use a simple approach
        conditions = [
            impact_metrics['risk_score'] == 0,
            impact_metrics['risk_score'] > 0
        ]
        choices = ['Low', 'High']
        impact_metrics['risk_category'] = np.select(conditions, choices, default='Medium')
    
    return impact_metrics

def analyze_seasonal_species_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze seasonal patterns in species activity
    """
    # Create season-species cross-tabulation
    seasonal_activity = pd.crosstab(
        df['INCIDENT_MONTH'],
        df['SPECIES'],
        normalize='columns'
    )
    
    # Identify peak seasons for each species
    peak_seasons = pd.DataFrame(index=seasonal_activity.columns)
    peak_seasons['peak_season'] = seasonal_activity.idxmax()
    peak_seasons['seasonal_concentration'] = seasonal_activity.max()
    
    # Calculate seasonal risk metrics
    seasonal_risk = df.groupby(['INCIDENT_MONTH', 'SPECIES']).agg({
        'DAMAGE_SCORE': 'mean',
        'TOTAL_COST': 'mean',
        'INDICATED_DAMAGE': 'mean'
    }).unstack()
    
    return {
        'seasonal_activity': seasonal_activity,
        'peak_seasons': peak_seasons,
        'seasonal_risk': seasonal_risk
    }

def analyze_species_strike_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns in how different species strike aircraft
    """
    # Define a safe mode function that handles empty results
    def safe_mode(x):
        mode_result = x.mode()
        if len(mode_result) > 0:
            return mode_result.iloc[0]
        return np.nan
    
    # Analyze strike locations
    strike_locations = pd.DataFrame()
    strike_cols = [col for col in df.columns if col.startswith('STR_')]
    
    for species in df['SPECIES'].unique():
        species_data = df[df['SPECIES'] == species]
        for col in strike_cols:
            strike_locations.loc[species, col] = species_data[col].mean()
    
    # Analyze height and speed patterns
    height_speed_patterns = df.groupby('SPECIES').agg({
        'HEIGHT': ['mean', 'std', lambda x: stats.mode(x, keepdims=True)[0][0] if len(x) > 0 else np.nan],
        'SPEED': ['mean', 'std', lambda x: stats.mode(x, keepdims=True)[0][0] if len(x) > 0 else np.nan]
    })
    
    # Flatten column names
    height_speed_patterns.columns = ['_'.join(col).strip() for col in height_speed_patterns.columns.values]
    
    # Calculate typical strike scenarios
    strike_scenarios = df.groupby('SPECIES').agg({
        'PHASE_OF_FLIGHT': safe_mode,
        'TIME_OF_DAY': safe_mode,
        'SKY': safe_mode,
        'HEIGHT': 'median',
        'SPEED': 'median'
    })
    
    return {
        'strike_locations': strike_locations,
        'height_speed_patterns': height_speed_patterns,
        'typical_scenarios': strike_scenarios
    }