"""
Report generation utilities for wildlife strikes analysis
"""
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from jinja2 import Template

class ReportGenerator:
    def __init__(self, output_dir: Union[str, Path]):
        """
        Initialize report generator with output directory
        """
        self.output_dir = Path(output_dir)
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        
        # Create output directories if they don't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize report sections
        self.sections = []
        self.insights = []
        self.recommendations = []
    
    def add_section(
        self,
        title: str,
        content: str,
        level: int = 1
    ) -> None:
        """
        Add a section to the report
        """
        self.sections.append({
            'title': title,
            'content': content,
            'level': level
        })
    
    def add_insight(
        self,
        category: str,
        description: str,
        metrics: Dict[str, Any],
        severity: str = 'medium'
    ) -> None:
        """
        Add an insight to the report
        """
        self.insights.append({
            'category': category,
            'description': description,
            'metrics': metrics,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def add_recommendation(
        self,
        title: str,
        description: str,
        impact: str,
        effort: str,
        priority: str,
        metrics: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Add a recommendation to the report
        """
        self.recommendations.append({
            'title': title,
            'description': description,
            'impact': impact,
            'effort': effort,
            'priority': priority,
            'metrics': metrics or {},
            'timestamp': datetime.now().isoformat()
        })
    
    def save_timeseries_plot(
        self,
        df: pd.DataFrame,
        value_column: str,
        title: str,
        filename: str,
        rolling_window: Optional[int] = None
    ) -> None:
        """
        Save time series plot
        """
        plt.figure(figsize=(12, 6))
        
        if rolling_window:
            rolling_avg = df[value_column].rolling(window=rolling_window).mean()
            plt.plot(df.index, rolling_avg, label=f'{rolling_window}-period Moving Average')
        
        plt.plot(df.index, df[value_column], alpha=0.5, label='Raw Data')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.savefig(
            self.figures_dir / f'{filename}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def save_distribution_plot(
        self,
        data: pd.Series,
        title: str,
        filename: str,
        bins: int = 50
    ) -> None:
        """
        Save distribution plot
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, bins=bins)
        plt.title(title)
        
        plt.savefig(
            self.figures_dir / f'{filename}.png',
            bbox_inches='tight',
            dpi=300
        )
        plt.close()
    
    def save_interactive_map(
        self,
        df: pd.DataFrame,
        lat_col: str,
        lon_col: str,
        title: str,
        filename: str,
        color_col: Optional[str] = None
    ) -> None:
        """
        Save interactive map visualization
        """
        import folium
        
        # Create base map
        center_lat = df[lat_col].mean()
        center_lon = df[lon_col].mean()
        m = folium.Map(location=[center_lat, center_lon], zoom_start=4)
        
        # Add points
        for idx, row in df.iterrows():
            popup_text = '<br>'.join([
                f'{col}: {row[col]}'
                for col in df.columns
                if col not in [lat_col, lon_col]
            ])
            
            folium.CircleMarker(
                location=[row[lat_col], row[lon_col]],
                radius=5,
                popup=popup_text,
                color='red' if color_col and row[color_col] > df[color_col].mean() else 'blue'
            ).add_to(m)
        
        # Save map
        m.save(self.figures_dir / f'{filename}.html')
    
    def save_summary_stats(
        self,
        df: pd.DataFrame,
        group_col: str,
        metrics: List[str],
        filename: str
    ) -> None:
        """
        Save summary statistics
        """
        summary = df.groupby(group_col)[metrics].agg(['mean', 'std', 'min', 'max'])
        summary.to_csv(self.tables_dir / f'{filename}.csv')
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary from insights and recommendations
        """
        high_priority_insights = [
            insight for insight in self.insights
            if insight['severity'] == 'high'
        ]
        
        high_priority_recommendations = [
            rec for rec in self.recommendations
            if rec['priority'] == 'high'
        ]
        
        template = Template('''
        # Executive Summary
        
        ## Key Findings
        {% for insight in insights %}
        * {{ insight.description }}
        {% endfor %}
        
        ## Critical Recommendations
        {% for rec in recommendations %}
        * {{ rec.title }}: {{ rec.description }}
        {% endfor %}
        ''')
        
        return template.render(
            insights=high_priority_insights,
            recommendations=high_priority_recommendations
        )
    
    def generate_full_report(self) -> str:
        """
        Generate full analysis report
        """
        template = Template('''
        # Wildlife Strikes Analysis Report
        Generated on {{ timestamp }}
        
        {{ executive_summary }}
        
        {% for section in sections %}
        {{ '#' * section.level }} {{ section.title }}
        
        {{ section.content }}
        {% endfor %}
        
        # Detailed Insights
        {% for insight in insights %}
        ## {{ insight.category }}
        * {{ insight.description }}
        * Severity: {{ insight.severity }}
        * Key Metrics:
        {% for metric, value in insight.metrics.items() %}
            * {{ metric }}: {{ value }}
        {% endfor %}
        {% endfor %}
        
        # Recommendations
        {% for rec in recommendations %}
        ## {{ rec.title }}
        * Description: {{ rec.description }}
        * Impact: {{ rec.impact }}
        * Effort: {{ rec.effort }}
        * Priority: {{ rec.priority }}
        {% if rec.metrics %}
        * Supporting Metrics:
        {% for metric, value in rec.metrics.items() %}
            * {{ metric }}: {{ value }}
        {% endfor %}
        {% endif %}
        {% endfor %}
        ''')
        
        report = template.render(
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            executive_summary=self.generate_executive_summary(),
            sections=self.sections,
            insights=self.insights,
            recommendations=self.recommendations
        )
        
        # Save report
        report_path = self.output_dir / 'full_report.md'
        report_path.write_text(report)
        
        return report
    
    def save_insights_json(self) -> None:
        """
        Save insights as JSON for further analysis
        """
        insights_path = self.output_dir / 'insights.json'
        with open(insights_path, 'w') as f:
            json.dump(self.insights, f, indent=2)
    
    def save_recommendations_json(self) -> None:
        """
        Save recommendations as JSON for further analysis
        """
        recommendations_path = self.output_dir / 'recommendations.json'
        with open(recommendations_path, 'w') as f:
            json.dump(self.recommendations, f, indent=2)

class ReportInsightGenerator:
    """
    Generate standardized insights from analysis results
    """
    def __init__(self, output_dir: Union[str, Path]):
        self.report_gen = ReportGenerator(output_dir)
        self.insights = []
    
    def analyze_temporal_patterns(
        self,
        trend_metrics: Dict[str, float],
        seasonal_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights from temporal analysis
        """
        insight = {
            'category': 'Temporal Patterns',
            'metrics': {
                'trend_coefficient': trend_metrics.get('trend_coefficient', 0),
                'seasonal_strength': seasonal_metrics.get('seasonal_strength', 0)
            },
            'severity': 'medium'
        }
        
        # Generate description based on metrics
        trend_direction = 'increasing' if insight['metrics']['trend_coefficient'] > 0 else 'decreasing'
        seasonal_strength = 'strong' if insight['metrics']['seasonal_strength'] > 0.5 else 'weak'
        
        insight['description'] = (
            f"Wildlife strikes show a {trend_direction} trend with {seasonal_strength} "
            f"seasonal patterns. Annual growth rate: {insight['metrics']['trend_coefficient']:.2%}"
        )
        
        self.insights.append(insight)
        return insight
    
    def analyze_risk_patterns(
        self,
        risk_metrics: pd.DataFrame,
        safety_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate insights from risk analysis
        """
        insight = {
            'category': 'Risk Assessment',
            'metrics': {
                'high_risk_airports': len(risk_metrics[risk_metrics['risk_score'] > risk_metrics['risk_score'].mean()]),
                'damage_rate': safety_metrics.get('damage_rate', 0)
            },
            'severity': 'high' if safety_metrics.get('damage_rate', 0) > 0.3 else 'medium'
        }
        
        insight['description'] = (
            f"Identified {insight['metrics']['high_risk_airports']} high-risk airports. "
            f"Overall damage rate: {insight['metrics']['damage_rate']:.2%}"
        )
        
        self.insights.append(insight)
        return insight
    
    def analyze_economic_impact(
        self,
        cost_metrics: Dict[str, float],
        trend_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights from economic analysis
        """
        insight = {
            'category': 'Economic Impact',
            'metrics': {
                'total_cost': cost_metrics['total_cost'],
                'mean_cost': cost_metrics['mean_cost'],
                'trend_increasing': trend_metrics['trend_increasing'],
                'annual_growth_rate': trend_metrics['annual_growth_rate']
            },
            'severity': 'high' if trend_metrics['trend_increasing'] else 'medium'
        }
        
        trend_description = (
            'increasing' if trend_metrics['trend_increasing']
            else 'decreasing'
        )
        
        insight['description'] = (
            f"Total economic impact: ${insight['metrics']['total_cost']:,.2f}. "
            f"Costs are {trend_description} at {insight['metrics']['annual_growth_rate']:.2%} annually"
        )
        
        self.insights.append(insight)
        return insight
    
    def analyze_species_patterns(
        self,
        species_metrics: pd.DataFrame,
        temporal_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Generate insights from species analysis
        """
        insight = {
            'category': 'Species Patterns',
            'metrics': {
                'high_risk_species': len(species_metrics[species_metrics['risk_score'] > species_metrics['risk_score'].mean()]),
                'seasonal_concentration': temporal_metrics['seasonal_concentration']
            },
            'severity': 'high' if temporal_metrics['seasonal_concentration'] > 0.7 else 'medium'
        }
        
        insight['description'] = (
            f"Identified {insight['metrics']['high_risk_species']} high-risk species. "
            f"Species activity shows {insight['metrics']['seasonal_concentration']:.2%} seasonal concentration"
        )
        
        self.insights.append(insight)
        return insight
    
    def analyze_operational_factors(
        self,
        operational_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate insights from operational analysis
        """
        insight = {
            'category': 'Operational Factors',
            'metrics': {
                'high_risk_phases': operational_metrics['high_risk_phases'],
                'warning_effectiveness': operational_metrics['warning_effectiveness']
            },
            'severity': 'high'
        }
        
        insight['description'] = (
            f"Highest risk during {', '.join(insight['metrics']['high_risk_phases'])}. "
            f"Warning systems {insight['metrics']['warning_effectiveness']:.2%} effective"
        )
        
        self.insights.append(insight)
        return insight
    
    def save_reports(self) -> None:
        """
        Save all insights and generate reports
        """
        # Add insights to report generator
        for insight in self.insights:
            self.report_gen.add_insight(
                category=insight['category'],
                description=insight['description'],
                metrics=insight['metrics'],
                severity=insight['severity']
            )
        
        # Generate and save reports
        self.report_gen.generate_full_report()
        self.report_gen.save_insights_json()