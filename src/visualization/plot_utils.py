"""
Visualization utilities for wildlife strikes analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Tuple, Optional, Union

def set_plotting_style() -> None:
    """
    Set consistent plotting style for all visualizations
    """
    plt.style.use('seaborn-v0_8')
    sns.set_palette('deep')
    plt.rcParams['figure.figsize'] = (12, 6)
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.dpi'] = 300

def plot_temporal_trends(
    df: pd.DataFrame,
    freq: str = 'M',
    rolling_window: Optional[int] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Plot temporal trends in wildlife strikes
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Create time series
    ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq=freq))['INDEX_NR'].count()
    
    # Plot raw data
    ax.plot(ts.index, ts.values, alpha=0.5, label='Raw Data')
    
    if rolling_window:
        # Add rolling average
        rolling = ts.rolling(window=rolling_window).mean()
        ax.plot(rolling.index, rolling.values, 
               linewidth=2, label=f'{rolling_window}-period Moving Average')
    
    ax.set_title('Wildlife Strikes Over Time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Strikes')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_geographic_distribution(
    df: pd.DataFrame,
    color_metric: Optional[str] = None,
    save_path: Optional[str] = None
) -> None:
    """
    Create interactive map of strike locations
    """
    center_lat = df['AIRPORT_LATITUDE'].mean()
    center_lon = df['AIRPORT_LONGITUDE'].mean()
    
    df = df.copy()
    df['TOTAL_COST'] = df['COST_REPAIRS_INFL_ADJ'].fillna(0) + df['COST_OTHER_INFL_ADJ'].fillna(0)
    
    if color_metric:
        fig = px.scatter_mapbox(
            df,
            lat='AIRPORT_LATITUDE',
            lon='AIRPORT_LONGITUDE',
            color=color_metric,
            title='Geographic Distribution of Wildlife Strikes',
            hover_data=['AIRPORT', 'TOTAL_COST', 'DAMAGE_LEVEL']
        )
    else:
        fig = px.scatter_mapbox(
            df,
            lat='AIRPORT_LATITUDE',
            lon='AIRPORT_LONGITUDE',
            title='Geographic Distribution of Wildlife Strikes',
            hover_data=['AIRPORT', 'TOTAL_COST', 'DAMAGE_LEVEL']
        )
    
    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox=dict(
            zoom=3,
            center=dict(lat=center_lat, lon=center_lon)
        )
    )
    
    if save_path:
        fig.write_html(save_path)

def plot_risk_heatmap(
    risk_matrix: pd.DataFrame,
    title: str,
    save_path: Optional[str] = None
) -> None:
    """
    Create risk heatmap visualization
    """
    plt.figure(figsize=(12, 10))
    
    sns.heatmap(
        risk_matrix,
        annot=True,
        cmap='YlOrRd',
        center=risk_matrix.values.mean(),
        fmt='.2f'
    )
    
    plt.title(title)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_species_distribution(
    df: pd.DataFrame,
    top_n: int = 10,
    save_path: Optional[str] = None
) -> None:
    """
    Plot distribution of wildlife species involved in strikes
    """
    species_counts = df['SPECIES'].value_counts().head(top_n)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=species_counts.values, y=species_counts.index)
    
    plt.title(f'Top {top_n} Species Involved in Wildlife Strikes')
    plt.xlabel('Number of Strikes')
    plt.ylabel('Species')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_cost_analysis(
    df: pd.DataFrame,
    bins: int = 50,
    save_path: Optional[str] = None
) -> None:
    """
    Plot cost distribution analysis
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    df = df.copy()
    df['TOTAL_COST'] = df['COST_REPAIRS_INFL_ADJ'].fillna(0) + df['COST_OTHER_INFL_ADJ'].fillna(0)
    
    # Plot histogram of costs
    sns.histplot(
        data=df,
        x='TOTAL_COST',
        bins=bins,
        ax=ax1
    )
    ax1.set_title('Distribution of Strike Costs')
    ax1.set_xlabel('Total Cost ($)')
    ax1.set_ylabel('Frequency')
    
    # Plot log-transformed costs
    sns.histplot(
        data=df,
        x='TOTAL_COST',
        bins=bins,
        ax=ax2
    )
    ax2.set_xscale('log')
    ax2.set_title('Log Distribution of Strike Costs')
    ax2.set_xlabel('Total Cost ($) - Log Scale')
    ax2.set_ylabel('Frequency')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_temporal_patterns(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot multiple temporal patterns
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Monthly pattern
    monthly_counts = df.groupby('INCIDENT_MONTH')['INDEX_NR'].count()
    sns.barplot(x=monthly_counts.index, y=monthly_counts.values, ax=ax1)
    ax1.set_title('Monthly Distribution')
    ax1.set_xlabel('Month')
    ax1.set_ylabel('Number of Strikes')
    
    # Time of day pattern
    time_counts = df.groupby('TIME_OF_DAY')['INDEX_NR'].count()
    sns.barplot(x=time_counts.index, y=time_counts.values, ax=ax2)
    ax2.set_title('Time of Day Distribution')
    ax2.set_xlabel('Time of Day')
    ax2.set_ylabel('Number of Strikes')
    
    # Yearly trend
    yearly_counts = df.groupby('INCIDENT_YEAR')['INDEX_NR'].count()
    ax3.plot(yearly_counts.index, yearly_counts.values)
    ax3.set_title('Yearly Trend')
    ax3.set_xlabel('Year')
    ax3.set_ylabel('Number of Strikes')
    
    # Phase of flight
    phase_counts = df.groupby('PHASE_OF_FLIGHT')['INDEX_NR'].count()
    sns.barplot(x=phase_counts.values, y=phase_counts.index, ax=ax4)
    ax4.set_title('Phase of Flight Distribution')
    ax4.set_xlabel('Number of Strikes')
    ax4.set_ylabel('Phase of Flight')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_damage_analysis(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Plot damage analysis visualizations
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Damage type distribution
    damage_counts = df['DAMAGE_LEVEL'].value_counts()
    sns.barplot(x=damage_counts.index, y=damage_counts.values, ax=ax1)
    ax1.set_title('Distribution of Damage Types')
    ax1.set_xlabel('Damage Category')
    ax1.set_ylabel('Number of Incidents')
    
    df = df.copy()
    df['TOTAL_COST'] = df['COST_REPAIRS_INFL_ADJ'].fillna(0) + df['COST_OTHER_INFL_ADJ'].fillna(0)
    
    # Cost by damage type
    sns.boxplot(data=df, x='DAMAGE_LEVEL', y='TOTAL_COST', ax=ax2)
    ax2.set_yscale('log')
    ax2.set_title('Cost Distribution by Damage Type')
    ax2.set_xlabel('Damage Category')
    ax2.set_ylabel('Total Cost ($) - Log Scale')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def create_summary_dashboard(
    df: pd.DataFrame,
    save_path: Optional[str] = None
) -> None:
    """
    Create a comprehensive dashboard of key metrics
    """
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(3, 3)
    
    # Time series trend
    ax1 = fig.add_subplot(gs[0, :])
    ts = df.groupby(pd.Grouper(key='INCIDENT_DATE', freq='M'))['INDEX_NR'].count()
    ax1.plot(ts.index, ts.values)
    ax1.set_title('Strike Incidents Over Time')
    
    # Species distribution
    ax2 = fig.add_subplot(gs[1, 0])
    species_counts = df['SPECIES'].value_counts().head(10)
    sns.barplot(x=species_counts.values, y=species_counts.index, ax=ax2)
    ax2.set_title('Top 10 Species')
    
    # Damage distribution
    ax3 = fig.add_subplot(gs[1, 1])
    damage_counts = df['DAMAGE_LEVEL'].value_counts()
    sns.barplot(x=damage_counts.index, y=damage_counts.values, ax=ax3)
    ax3.set_title('Damage Distribution')
    
    df = df.copy()
    df['TOTAL_COST'] = df['COST_REPAIRS_INFL_ADJ'].fillna(0) + df['COST_OTHER_INFL_ADJ'].fillna(0)
    
    # Cost distribution
    ax4 = fig.add_subplot(gs[1, 2])
    sns.histplot(data=df, x='TOTAL_COST', ax=ax4)
    ax4.set_xscale('log')
    ax4.set_title('Cost Distribution')
    
    # Phase of flight
    ax5 = fig.add_subplot(gs[2, :])
    phase_counts = df.groupby('PHASE_OF_FLIGHT')['INDEX_NR'].count()
    sns.barplot(x=phase_counts.index, y=phase_counts.values, ax=ax5)
    ax5.set_title('Strikes by Phase of Flight')
    ax5.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_correlation_analysis(
    df: pd.DataFrame,
    variables: List[str],
    save_path: Optional[str] = None
) -> None:
    """
    Plot correlation analysis for selected variables
    """
    # Calculate correlations
    corr_matrix = df[variables].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap='coolwarm',
        center=0,
        fmt='.2f'
    )
    plt.title('Correlation Analysis')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_risk_factors(
    df: pd.DataFrame,
    risk_scores: pd.Series,
    save_path: Optional[str] = None
) -> None:
    """
    Plot analysis of risk factors
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot risk score distribution
    sns.histplot(data=risk_scores, ax=ax1)
    ax1.set_title('Distribution of Risk Scores')
    ax1.set_xlabel('Risk Score')
    ax1.set_ylabel('Frequency')
    
    # Plot top risk factors
    top_risks = risk_scores.nlargest(10)
    sns.barplot(x=top_risks.values, y=top_risks.index, ax=ax2)
    ax2.set_title('Top 10 Risk Factors')
    ax2.set_xlabel('Risk Score')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()