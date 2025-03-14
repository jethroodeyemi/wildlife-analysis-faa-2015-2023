# %% [markdown]
# # FAA Wildlife Strikes Analysis (1990-2023)
# ## Comprehensive Analysis for Safety and Risk Assessment

# %% [markdown]
# ## Table of Contents
# 1. Introduction and Data Overview
# 2. Temporal Analysis
#    - Long-term Trends
#    - Seasonal Patterns
#    - Time-of-Day Analysis
# 3. Geographic Analysis
#    - Regional Distribution
#    - Airport-specific Patterns
#    - Risk Mapping
# 4. Species Analysis
#    - Most Common Species
#    - High-Risk Species
#    - Strike Patterns by Species
# 5. Impact Analysis
#    - Economic Impact
#    - Aircraft Damage Patterns
#    - Operational Disruption
# 6. Risk Assessment
#    - Risk Factors
#    - Predictive Indicators
#    - Mitigation Recommendations
# 7. Conclusions and Recommendations

# %% [markdown]
# ## Setup and Configuration

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Import custom modules
from src.preprocessing.data_cleaner import process_wildlife_data
from src.visualization.plot_utils import *
from src.statistics.statistical_analysis import *
from src.analysis.temporal_patterns import *
from src.analysis.spatial_patterns import *
from src.analysis.wildlife_behavior import *
from src.analysis.economic_impact import *
from src.analysis.operational_impact import *
from src.analysis.report_insights import *
from utils.helpers.report_generators import ReportGenerator
from utils.validators.data_validators import generate_validation_report

# Configure output paths
OUTPUT_DIR = Path("reports")
FIGURE_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Set up visualization defaults
set_plotting_style()

# %% [markdown]
# ## Data Loading and Validation

# %%
# Load the dataset
print("Loading dataset...")
df = pd.read_csv('FAA_Wildlife_Strikes_Data_03-30-2025.csv')

# Validate data quality
print("\nValidating data quality...")
validation_report = generate_validation_report(df)
total_records = validation_report['summary']['total_records']

# Calculate invalid records by summing all validation issues
invalid_count = sum(
    len(issues) if isinstance(issues, list) else 
    len(issues.keys()) if isinstance(issues, dict) else 0
    for key, issues in validation_report.items()
    if key != 'summary'
)

print(f"Total records: {total_records:,}")
print(f"Invalid records: {invalid_count:,}")

# Process and clean data
print("\nProcessing and cleaning data...")
df_clean = process_wildlife_data(df)

# %% [markdown]
# ## Temporal Analysis

# %%
print("Analyzing temporal patterns...")

# Analyze long-term trends
trend_results = analyze_long_term_trends(df_clean)
print("\nTrend Analysis Results:")
for metric, results in trend_results.items():
    print(f"\n{metric}:")
    print(f"Trend coefficient: {results['trend_coefficient']:.3f}")
    print(f"Significant: {results['is_significant']}")

# Analyze cyclical patterns
cyclical_results = analyze_cyclical_patterns(df_clean)
print("\nSeasonality Strength:", cyclical_results['seasonal_strength'])

# Analyze monthly patterns
monthly_patterns = analyze_monthly_patterns(df_clean)
print("\nPeak Months:")
for metric, month in monthly_patterns['peak_months'].items():
    print(f"{metric}: Month {month}")

# %% [markdown]
# ## Geographic Analysis

# %%
print("Analyzing spatial patterns...")

# Analyze airport risk factors
airport_risks = analyze_airport_risk_factors(df_clean)
print("\nTop 10 Highest Risk Airports:")
print(airport_risks.nlargest(10, 'risk_score')[['incident_rate', 'risk_score']])

# Identify spatial clusters
cluster_metrics = identify_spatial_clusters(df_clean)
print(f"\nIdentified {len(cluster_metrics)} distinct spatial clusters")

# Analyze regional patterns
regional_results = analyze_regional_patterns(df_clean)
print("\nTop 5 Regions by Risk Score:")
print(regional_results['regional'].nlargest(5, 'risk_score'))

# %% [markdown]
# ## Species Analysis

# %%
print("Analyzing species behavior and impact...")

# Analyze species behavior patterns
species_behavior = analyze_species_behavior(df_clean)
print("\nSpecies Behavior Analysis:")
print(f"Total unique species: {len(species_behavior)}")

# Identify species clusters
species_metrics, cluster_profiles = identify_species_clusters(df_clean)
print("\nSpecies Clusters:")
for cluster, profile in cluster_profiles.items():
    print(f"\n{cluster}:")
    print(f"Size: {profile['size']}")
    print(f"Average height: {profile['avg_height']:.1f}")

# Analyze species impact
impact_metrics = analyze_species_impact(df_clean)
print("\nTop 5 Most Impactful Species:")
print(impact_metrics.nlargest(5, 'risk_score')[['risk_score', 'risk_category']])

# %% [markdown]
# ## Economic Impact Analysis

# %%
print("Analyzing economic impact...")

# Initialize economic impact analyzer
economic_analyzer = EconomicImpactAnalyzer(df_clean)

# Analyze cost trends
cost_trends = economic_analyzer.analyze_cost_trends()
print("\nCost Trend Analysis:")
print(f"Trend coefficient: {cost_trends['trend_coefficient']:.2f}")
print(f"R-squared: {cost_trends['r_squared']:.3f}")

# Analyze cost distribution
cost_dist = economic_analyzer.analyze_cost_distribution()
print("\nCost Distribution Statistics:")
print(f"Mean cost: ${cost_dist['percentiles']['percentile_50']:,.2f}")
print(f"Median cost: ${cost_dist['percentiles']['percentile_50']:,.2f}")
print(f"95th percentile: ${cost_dist['percentiles']['percentile_95']:,.2f}")

# Calculate risk-adjusted costs and forecasts
summary_report = economic_analyzer.generate_summary_report()
risk_costs = summary_report['basic_metrics']
print("\nRisk-Adjusted Cost Analysis:")
print(f"Total cost: ${risk_costs['total_cost']:,.2f}")
print(f"Cost per incident: ${risk_costs['cost_per_incident']:,.2f}")

# Forecast future costs
forecast_results = economic_analyzer.forecast_future_costs()
forecast_mean = forecast_results[0]
print("\nCost Forecast (12 months):")
print(forecast_mean)

# %% [markdown]
# ## Operational Impact Analysis

# %%
print("Analyzing operational impacts...")

# Analyze operational delays
delay_metrics = analyze_operational_delays(df_clean)
print("\nOperational Delay Analysis:")
print(f"Average out-of-service time: {delay_metrics['average_out_of_service_hours']:.1f} hours")
print(f"Total impact hours: {delay_metrics['total_impact_hours']:,.0f}")

# Analyze phase of flight risk
phase_risk = analyze_phase_of_flight_risk(df_clean)
print("\nHighest Risk Flight Phases:")
print(phase_risk.nlargest(3, 'risk_score'))

# Analyze aircraft vulnerability
vulnerability_metrics, critical_components = analyze_aircraft_vulnerability(df_clean)
print("\nCritical Components:")
print("High damage rate:", critical_components['high_damage_rate'])
print("High repair cost:", critical_components['high_repair_cost'])

# Calculate safety metrics
safety_stats = calculate_safety_metrics(df_clean)
print("\nSafety Metrics:")
for metric, value in safety_stats.items():
    print(f"{metric}: {value:.3f}")

# %% [markdown]
# ## Generate Insights and Recommendations

# %%
# Initialize report generator
insight_generator = ReportInsightGenerator(OUTPUT_DIR)

# Generate temporal insights
temporal_insight = insight_generator.analyze_temporal_patterns(
    trend_results['incident_count'],
    cyclical_results
)

# Generate risk insights
risk_insight = insight_generator.analyze_risk_patterns(
    airport_risks,
    {'damage_rate': safety_stats['damage_rate']}
)

# Generate economic insights
economic_insight = insight_generator.analyze_economic_impact(
    {'total_cost': cost_dist['mean'] * len(df_clean),
     'mean_cost': cost_dist['mean']},
    {'trend_increasing': cost_trends['trend_coefficient'] > 0,
     'annual_growth_rate': cost_trends['trend_coefficient']}
)

# Generate species insights
species_insight = insight_generator.analyze_species_patterns(
    impact_metrics,
    {'seasonal_concentration': cyclical_results['seasonal_strength']}
)

# Generate operational insights
operational_insight = insight_generator.analyze_operational_factors(
    {'high_risk_phases': phase_risk.nlargest(3, 'risk_score').index.tolist(),
     'warning_effectiveness': 0.7}  # Example value
)

# Add insights to generator
insight_generator.insights.extend([
    temporal_insight,
    risk_insight,
    economic_insight,
    species_insight,
    operational_insight
])

# Generate and save reports
print("\nGenerating reports...")
insight_generator.save_reports()

# %% [markdown]
# ## Save Visualizations and Tables

# %%
# Initialize report generator for visualizations
report_gen = ReportGenerator(OUTPUT_DIR)

# Save time series plots
report_gen.save_timeseries_plot(
    df_clean,
    'INDEX NR',
    'Wildlife Strikes Trend (1990-2023)',
    'strikes_trend',
    rolling_window=12
)

# Save cost distribution
report_gen.save_distribution_plot(
    df_clean['TOTAL_COST'],
    'Distribution of Strike Costs',
    'cost_distribution'
)

# Save geographic visualization
report_gen.save_interactive_map(
    df_clean,
    'AIRPORT_LATITUDE',
    'AIRPORT_LONGITUDE',
    'Wildlife Strike Geographic Distribution',
    'strike_distribution',
    color_col='DAMAGE_SCORE'
)

# Save summary statistics
report_gen.save_summary_stats(
    df_clean,
    group_col='SPECIES',
    metrics=['TOTAL_COST', 'DAMAGE_SCORE', 'HEIGHT', 'SPEED'],
    filename='species_summary'
)

print("\nAnalysis complete. Reports and visualizations have been saved to the 'reports' directory.")