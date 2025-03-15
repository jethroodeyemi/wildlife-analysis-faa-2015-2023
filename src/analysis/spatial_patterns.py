"""
Geographic and spatial analysis for wildlife strikes
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import geopandas as gpd
from shapely.geometry import Point


def analyze_airport_risk_factors(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze risk factors specific to airports
    """
    airport_metrics = df.groupby('AIRPORT').agg({
        'INDEX_NR': 'count',
        'INDICATED_DAMAGE': 'mean',
        'TOTAL_COST': ['sum', 'mean'],
        'AIRPORT_LATITUDE': 'first',
        'AIRPORT_LONGITUDE': 'first'
    })
    
    # Flatten column names
    airport_metrics.columns = ['_'.join(col).strip() for col in airport_metrics.columns.values]
    
    # Calculate incident rate (normalize by total incidents)
    total_incidents = len(df)
    airport_metrics['incident_rate'] = airport_metrics['INDEX_NR_count'] / total_incidents
    
    # Calculate risk score
    airport_metrics['risk_score'] = (
        airport_metrics['INDICATED_DAMAGE_mean'] * 
        np.log1p(airport_metrics['TOTAL_COST_mean'])
    )
    
    return airport_metrics

def identify_spatial_clusters(
    df: pd.DataFrame,
    eps_km: float = 50,
    min_samples: int = 5
) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """
    Identify spatial clusters of wildlife strikes
    """
    # Create coordinates array and filter out rows with missing coordinates
    df_valid = df.dropna(subset=['AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE'])
    
    # Check if we have enough valid data points to perform clustering
    if len(df_valid) < min_samples:
        print(f"Warning: Not enough valid coordinates for clustering. Found {len(df_valid)} valid points.")
        return pd.DataFrame()
        
    coords = df_valid[['AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE']].values
    
    # Perform DBSCAN clustering
    db = DBSCAN(
        eps=eps_km/111.32,  # Convert km to degrees (approximate)
        min_samples=min_samples,
        metric='haversine'
    ).fit(np.radians(coords))
    
    # Add cluster labels to dataframe
    df_clustered = df_valid.copy()
    df_clustered['cluster'] = db.labels_
    
    # Analyze clusters
    cluster_metrics = df_clustered[df_clustered['cluster'] != -1].groupby('cluster').agg({
        'INDEX_NR': 'count',
        'TOTAL_COST': 'sum',
        'AIRPORT_LATITUDE': 'mean',
        'AIRPORT_LONGITUDE': 'mean',
        'AIRPORT': lambda x: list(x.unique())
    })
    
    return cluster_metrics

def analyze_regional_patterns(
    df: pd.DataFrame
) -> Dict[str, pd.DataFrame]:
    """
    Analyze patterns by FAA region and state
    """
    # Regional analysis
    regional_metrics = df.groupby('FAAREGION').agg({
        'INDEX_NR': 'count',
        'TOTAL_COST': ['sum', 'mean'],
        'INDICATED_DAMAGE': 'mean'
    })
    
    # State analysis
    state_metrics = df.groupby('STATE').agg({
        'INDEX_NR': 'count',
        'TOTAL_COST': ['sum', 'mean'],
        'INDICATED_DAMAGE': 'mean'
    })
    
    # Calculate regional risk scores
    for metrics in [regional_metrics, state_metrics]:
        metrics.columns = ['_'.join(col).strip() for col in metrics.columns.values]
        metrics['risk_score'] = (
            metrics['INDICATED_DAMAGE_mean'] * 
            np.log1p(metrics['TOTAL_COST_mean'])
        )
    
    return {
        'regional': regional_metrics,
        'state': state_metrics
    }

def calculate_spatial_correlations(
    df: pd.DataFrame,
    max_distance_km: float = 1000
) -> Dict[str, pd.DataFrame]:
    """
    Calculate spatial correlations between incidents
    """
    # Filter rows with missing coordinates or metrics
    df_valid = df.dropna(subset=['AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE', 'TOTAL_COST', 'INDICATED_DAMAGE'])
    
    # Check if we have enough valid data points
    if len(df_valid) < 2:
        print("Warning: Not enough valid data points for spatial correlation analysis.")
        return {
            'distance_matrix': pd.DataFrame(),
            'correlations': pd.DataFrame()
        }
    
    # Create spatial index
    df_valid['geometry'] = df_valid.apply(
        lambda row: Point(row['AIRPORT_LONGITUDE'], row['AIRPORT_LATITUDE']),
        axis=1
    )
    gdf = gpd.GeoDataFrame(df_valid, geometry='geometry')
    
    # Calculate distance matrix
    coords = df_valid[['AIRPORT_LATITUDE', 'AIRPORT_LONGITUDE']].values
    distances = pdist(coords, metric=lambda u, v: geodesic(u, v).km)
    dist_matrix = squareform(distances)
    
    # Calculate correlation metrics for nearby airports
    nearby_correlations = pd.DataFrame()
    
    for metric in ['TOTAL_COST', 'INDICATED_DAMAGE']:
        values = df_valid[metric].values
        # Calculate correlation for pairs within max_distance
        mask = dist_matrix <= max_distance_km
        np.fill_diagonal(mask, False)  # Exclude self-correlations
        
        if mask.any():
            corr = np.corrcoef(
                values[mask.any(axis=1)],
                values[mask.any(axis=0)]
            )[0, 1]
            
            nearby_correlations.loc[metric, 'spatial_correlation'] = corr
    
    return {
        'distance_matrix': pd.DataFrame(
            dist_matrix,
            index=df_valid['AIRPORT'],
            columns=df_valid['AIRPORT']
        ),
        'correlations': nearby_correlations
    }

def analyze_route_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analyze patterns in flight routes and enroute incidents
    """
    # Extract enroute incidents
    enroute_incidents = df[df['ENROUTE_STATE'].notna()]
    
    # Analyze patterns by route
    route_metrics = enroute_incidents.groupby('ENROUTE_STATE').agg({
        'INDEX_NR': 'count',
        'TOTAL_COST': ['sum', 'mean'],
        'INDICATED_DAMAGE': 'mean',
        'HEIGHT': 'mean',
        'SPEED': 'mean'
    })
    
    # Flatten column names
    route_metrics.columns = ['_'.join(col).strip() for col in route_metrics.columns.values]
    
    # Calculate risk score for routes
    route_metrics['risk_score'] = (
        route_metrics['INDICATED_DAMAGE_mean'] * 
        np.log1p(route_metrics['TOTAL_COST_mean'])
    )
    
    return route_metrics

def identify_high_risk_zones(
    df: pd.DataFrame,
    risk_threshold: float = 0.75
) -> Dict[str, pd.DataFrame]:
    """
    Identify and analyze high-risk geographical zones
    """
    # Calculate base risk metrics
    airport_risks = analyze_airport_risk_factors(df)
    
    # Identify high-risk airports
    high_risk_threshold = airport_risks['risk_score'].quantile(risk_threshold)
    high_risk_airports = airport_risks[airport_risks['risk_score'] > high_risk_threshold]
    
    # Analyze characteristics of high-risk zones
    high_risk_zones = df[df['AIRPORT'].isin(high_risk_airports.index)].groupby('AIRPORT').agg({
        'SPECIES': lambda x: x.value_counts().index[0],  # Most common species
        'TIME_OF_DAY': lambda x: x.value_counts().index[0],  # Most common time
        'PHASE_OF_FLIGHT': lambda x: x.value_counts().index[0],  # Most common flight phase
        'HEIGHT': 'mean',
        'SPEED': 'mean',
        'TOTAL_COST': 'sum'
    })
    
    return {
        'high_risk_airports': high_risk_airports,
        'zone_characteristics': high_risk_zones
    }