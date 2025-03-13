"""
Economic impact analysis for wildlife strikes
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose

class EconomicImpactAnalyzer:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with wildlife strikes dataset
        """
        self.df = df
        self._prepare_data()
    
    def _prepare_data(self) -> None:
        """
        Prepare data for economic analysis
        """
        # Convert cost columns to numeric, replacing non-numeric values with NaN
        cost_columns = ['COST_REPAIRS', 'COST_OTHER', 'COST_TOTAL']
        for col in cost_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
        
        # Create monthly and yearly aggregations
        if 'INCIDENT_DATE' in self.df.columns:
            self.df['INCIDENT_DATE'] = pd.to_datetime(self.df['INCIDENT_DATE'])
            self.monthly_costs = self.df.groupby(
                pd.Grouper(key='INCIDENT_DATE', freq='M')
            )['COST_TOTAL'].sum()
            self.yearly_costs = self.df.groupby(
                pd.Grouper(key='INCIDENT_DATE', freq='Y')
            )['COST_TOTAL'].sum()
    
    def calculate_basic_metrics(self) -> Dict[str, float]:
        """
        Calculate basic economic impact metrics
        """
        metrics = {
            'total_cost': self.df['COST_TOTAL'].sum(),
            'mean_cost': self.df['COST_TOTAL'].mean(),
            'median_cost': self.df['COST_TOTAL'].median(),
            'max_cost': self.df['COST_TOTAL'].max(),
            'std_cost': self.df['COST_TOTAL'].std(),
            'total_incidents': len(self.df),
            'incidents_with_cost': self.df['COST_TOTAL'].notna().sum(),
            'cost_per_incident': self.df['COST_TOTAL'].sum() / len(self.df)
        }
        
        return metrics
    
    def analyze_cost_trends(self) -> Dict[str, Any]:
        """
        Analyze trends in cost data
        """
        # Prepare time series data
        monthly_series = self.monthly_costs.fillna(0)
        
        # Perform time series decomposition
        decomposition = seasonal_decompose(
            monthly_series,
            period=12,
            extrapolate_trend='freq'
        )
        
        # Calculate trend
        X = np.arange(len(monthly_series)).reshape(-1, 1)
        y = monthly_series.values
        model = sm.OLS(y, sm.add_constant(X)).fit()
        
        # Calculate year-over-year growth
        yoy_growth = (
            (self.yearly_costs - self.yearly_costs.shift(1))
            / self.yearly_costs.shift(1)
        ).mean()
        
        return {
            'trend_coefficient': model.params[1],
            'trend_pvalue': model.pvalues[1],
            'trend_increasing': model.params[1] > 0,
            'annual_growth_rate': yoy_growth,
            'r_squared': model.rsquared,
            'seasonal_strength': np.std(decomposition.seasonal) / np.std(monthly_series)
        }
    
    def analyze_cost_distribution(self) -> Dict[str, Any]:
        """
        Analyze the distribution of costs
        """
        costs = self.df['COST_TOTAL'].dropna()
        
        # Fit distribution
        shape, loc, scale = stats.lognorm.fit(costs[costs > 0])
        
        # Calculate percentiles
        percentiles = {
            f'percentile_{p}': np.percentile(costs, p)
            for p in [25, 50, 75, 90, 95, 99]
        }
        
        return {
            'distribution_params': {
                'shape': shape,
                'location': loc,
                'scale': scale
            },
            'percentiles': percentiles,
            'skewness': stats.skew(costs),
            'kurtosis': stats.kurtosis(costs)
        }
    
    def analyze_cost_factors(self) -> Dict[str, Any]:
        """
        Analyze factors affecting cost
        """
        factors = {}
        
        # Analyze cost by aircraft size
        if 'AIRCRAFT_MASS' in self.df.columns:
            factors['cost_by_aircraft_size'] = (
                self.df.groupby('AIRCRAFT_MASS')['COST_TOTAL']
                .agg(['mean', 'count', 'sum'])
                .to_dict()
            )
        
        # Analyze cost by strike location
        if 'STRIKE_LOCATION' in self.df.columns:
            factors['cost_by_location'] = (
                self.df.groupby('STRIKE_LOCATION')['COST_TOTAL']
                .agg(['mean', 'count', 'sum'])
                .to_dict()
            )
        
        # Analyze cost by flight phase
        if 'FLIGHT_PHASE' in self.df.columns:
            factors['cost_by_phase'] = (
                self.df.groupby('FLIGHT_PHASE')['COST_TOTAL']
                .agg(['mean', 'count', 'sum'])
                .to_dict()
            )
        
        return factors
    
    def get_high_impact_incidents(
        self,
        threshold_percentile: float = 95
    ) -> pd.DataFrame:
        """
        Get details of high-impact incidents
        """
        threshold = np.percentile(
            self.df['COST_TOTAL'].dropna(),
            threshold_percentile
        )
        
        return self.df[self.df['COST_TOTAL'] >= threshold].copy()
    
    def calculate_regional_impact(
        self,
        region_column: str = 'STATE'
    ) -> pd.DataFrame:
        """
        Calculate economic impact by region
        """
        if region_column not in self.df.columns:
            raise ValueError(f"Column {region_column} not found in dataset")
        
        return (
            self.df.groupby(region_column)
            .agg({
                'COST_TOTAL': ['sum', 'mean', 'count'],
                'INCIDENT_DATE': 'max'  # Most recent incident
            })
            .round(2)
            .sort_values(('COST_TOTAL', 'sum'), ascending=False)
        )
    
    def forecast_future_costs(
        self,
        periods: int = 12
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Forecast future costs using time series analysis
        """
        # Prepare data
        y = self.monthly_costs.fillna(0)
        
        # Fit model
        model = sm.tsa.SARIMAX(
            y,
            order=(1, 1, 1),
            seasonal_order=(1, 1, 1, 12)
        )
        results = model.fit()
        
        # Generate forecast
        forecast = results.get_forecast(steps=periods)
        
        return (
            forecast.predicted_mean,
            forecast.conf_int().iloc[:, 0],  # Lower bound
            forecast.conf_int().iloc[:, 1]   # Upper bound
        )
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive economic impact summary
        """
        basic_metrics = self.calculate_basic_metrics()
        trends = self.analyze_cost_trends()
        distribution = self.analyze_cost_distribution()
        factors = self.analyze_cost_factors()
        
        forecast_mean, forecast_lower, forecast_upper = self.forecast_future_costs()
        
        return {
            'basic_metrics': basic_metrics,
            'trends': trends,
            'distribution': distribution,
            'impact_factors': factors,
            'forecast': {
                'mean': forecast_mean.to_dict(),
                'lower_bound': forecast_lower.to_dict(),
                'upper_bound': forecast_upper.to_dict()
            }
        }