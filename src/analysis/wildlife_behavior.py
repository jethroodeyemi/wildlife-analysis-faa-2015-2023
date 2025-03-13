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
    species_behavior = df.groupby('SPECIES').agg({
        'INDEX NR': 'count',
        'HEIGHT': ['mean', 'std', 'median'],
        'SPEED': ['mean', 'std'],
        'TIME_OF_DAY': lambda x: x.mode().iloc[0],
        'SEASON': lambda x: x.mode().iloc[0],
        'PRECIP': lambda x: x.mode().iloc[0],
        'SKY': lambda x: x.mode().iloc[0]
    })
    
    # Flatten column names
    species_behavior.columns = ['_'.join(col).strip() for col in species_behavior.columns.values]
    
    # Calculate frequency metrics
    total_incidents = len(df)
    species_behavior['frequency'] = species_behavior['INDEX NR_count'] / total_incidents
    
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

def analyze_species_impact(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze the impact of different species on aviation safety
    """
    # Calculate impact metrics
    impact_metrics = df.groupby('SPECIES').agg({
        'INDEX NR': 'count',
        'DAMAGE_SCORE': ['mean', 'std'],
        'TOTAL_COST': ['sum', 'mean'],
        'HAS_DAMAGE': 'mean',
        'NR_INJURIES': 'sum',
        'NR_FATALITIES': 'sum'
    })
    
    # Flatten column names
    impact_metrics.columns = ['_'.join(col).strip() for col in impact_metrics.columns.values]
    
    # Calculate risk score
    impact_metrics['risk_score'] = (
        impact_metrics['DAMAGE_SCORE_mean'] * 
        impact_metrics['HAS_DAMAGE_mean'] * 
        np.log1p(impact_metrics['TOTAL_COST_mean'])
    )
    
    # Categorize species by risk level
    risk_categories = pd.qcut(
        impact_metrics['risk_score'],
        q=4,
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    impact_metrics['risk_category'] = risk_categories
    
    return impact_metrics

def analyze_seasonal_species_patterns(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Analyze seasonal patterns in species activity
    """
    # Create season-species cross-tabulation
    seasonal_activity = pd.crosstab(
        df['SEASON'],
        df['SPECIES'],
        normalize='columns'
    )
    
    # Identify peak seasons for each species
    peak_seasons = pd.DataFrame(index=seasonal_activity.columns)
    peak_seasons['peak_season'] = seasonal_activity.idxmax()
    peak_seasons['seasonal_concentration'] = seasonal_activity.max()
    
    # Calculate seasonal risk metrics
    seasonal_risk = df.groupby(['SEASON', 'SPECIES']).agg({
        'DAMAGE_SCORE': 'mean',
        'TOTAL_COST': 'mean',
        'HAS_DAMAGE': 'mean'
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
    # Analyze strike locations
    strike_locations = pd.DataFrame()
    strike_cols = [col for col in df.columns if col.startswith('STR_')]
    
    for species in df['SPECIES'].unique():
        species_data = df[df['SPECIES'] == species]
        for col in strike_cols:
            strike_locations.loc[species, col] = species_data[col].mean()
    
    # Analyze height and speed patterns
    height_speed_patterns = df.groupby('SPECIES').agg({
        'HEIGHT': ['mean', 'std', lambda x: stats.mode(x, keepdims=True)[0][0]],
        'SPEED': ['mean', 'std', lambda x: stats.mode(x, keepdims=True)[0][0]]
    })
    
    # Flatten column names
    height_speed_patterns.columns = ['_'.join(col).strip() for col in height_speed_patterns.columns.values]
    
    # Calculate typical strike scenarios
    strike_scenarios = df.groupby('SPECIES').agg({
        'PHASE_OF_FLT': lambda x: x.mode().iloc[0],
        'TIME_OF_DAY': lambda x: x.mode().iloc[0],
        'SKY': lambda x: x.mode().iloc[0],
        'HEIGHT': 'median',
        'SPEED': 'median'
    })
    
    return {
        'strike_locations': strike_locations,
        'height_speed_patterns': height_speed_patterns,
        'typical_scenarios': strike_scenarios
    }