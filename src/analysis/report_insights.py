"""
Generation of insights and recommendations from wildlife strike analysis
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class InsightCategory:
    title: str
    findings: List[str]
    recommendations: List[str]
    priority: str  # High, Medium, Low
    impact_areas: List[str]
    supporting_metrics: Dict[str, float]

class ReportInsightGenerator:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.insights = []
    
    def analyze_temporal_patterns(
        self,
        trend_results: Dict,
        seasonal_patterns: Dict,
        significance_level: float = 0.05
    ) -> InsightCategory:
        """
        Generate insights from temporal analysis
        """
        findings = []
        recommendations = []
        
        # Analyze trend significance
        if trend_results['trend_coefficient'] > 0:
            growth_rate = trend_results['yoy_growth'] * 100
            findings.append(
                f"Significant increasing trend in wildlife strikes "
                f"(growth rate: {growth_rate:.1f}% per year)"
            )
            recommendations.append(
                "Enhance wildlife monitoring and detection systems to address "
                "increasing incident rates"
            )
        
        # Analyze seasonality
        if seasonal_patterns['seasonal_strength'] > 0.3:
            findings.append(
                "Strong seasonal pattern in wildlife strikes, with peak activity "
                "during migration seasons"
            )
            recommendations.append(
                "Adjust wildlife management strategies seasonally, with increased "
                "resources during peak periods"
            )
        
        return InsightCategory(
            title="Temporal Patterns",
            findings=findings,
            recommendations=recommendations,
            priority="High" if trend_results['significant'] else "Medium",
            impact_areas=["Safety", "Operations", "Resource Planning"],
            supporting_metrics={
                'trend_coefficient': trend_results['trend_coefficient'],
                'seasonal_strength': seasonal_patterns['seasonal_strength']
            }
        )
    
    def analyze_risk_patterns(
        self,
        risk_metrics: pd.DataFrame,
        damage_analysis: Dict,
        threshold: float = 0.75
    ) -> InsightCategory:
        """
        Generate insights from risk analysis
        """
        findings = []
        recommendations = []
        
        # Identify high-risk categories
        high_risk = risk_metrics[risk_metrics['risk_score'] > 
                               risk_metrics['risk_score'].quantile(threshold)]
        
        if not high_risk.empty:
            findings.append(
                f"Identified {len(high_risk)} high-risk categories requiring "
                "immediate attention"
            )
            recommendations.append(
                "Implement targeted risk mitigation strategies for identified "
                "high-risk categories"
            )
        
        # Analyze damage patterns
        if damage_analysis['damage_rate'] > 0.2:  # 20% threshold
            findings.append(
                f"High damage rate ({damage_analysis['damage_rate']*100:.1f}%) "
                "indicates need for improved prevention measures"
            )
            recommendations.append(
                "Enhance aircraft protection systems and operational procedures "
                "in high-risk scenarios"
            )
        
        return InsightCategory(
            title="Risk Patterns",
            findings=findings,
            recommendations=recommendations,
            priority="High",
            impact_areas=["Safety", "Risk Management", "Aircraft Protection"],
            supporting_metrics={
                'high_risk_categories': len(high_risk),
                'damage_rate': damage_analysis['damage_rate']
            }
        )
    
    def analyze_economic_impact(
        self,
        cost_analysis: Dict,
        forecast_results: Dict
    ) -> InsightCategory:
        """
        Generate insights from economic analysis
        """
        findings = []
        recommendations = []
        
        # Analyze current costs
        total_cost = cost_analysis['total_cost']
        avg_cost = cost_analysis['mean_cost']
        
        findings.append(
            f"Total economic impact of ${total_cost:,.2f} with average cost "
            f"per incident of ${avg_cost:,.2f}"
        )
        
        # Analyze cost trends
        if forecast_results['trend_increasing']:
            growth_rate = forecast_results['annual_growth_rate'] * 100
            findings.append(
                f"Projected cost increase of {growth_rate:.1f}% annually over "
                "next 5 years"
            )
            recommendations.append(
                "Develop comprehensive cost management strategy to address "
                "rising economic impact"
            )
        
        return InsightCategory(
            title="Economic Impact",
            findings=findings,
            recommendations=recommendations,
            priority="High" if total_cost > 1000000 else "Medium",
            impact_areas=["Financial", "Risk Management", "Budget Planning"],
            supporting_metrics={
                'total_cost': total_cost,
                'average_cost': avg_cost,
                'projected_growth': forecast_results.get('annual_growth_rate', 0)
            }
        )
    
    def analyze_species_patterns(
        self,
        species_analysis: pd.DataFrame,
        behavior_patterns: Dict
    ) -> InsightCategory:
        """
        Generate insights from species analysis
        """
        findings = []
        recommendations = []
        
        # Identify most problematic species
        top_species = species_analysis.nlargest(5, 'risk_score')
        
        findings.append(
            f"Top 5 highest-risk species account for "
            f"{(top_species['incident_rate'].sum()*100):.1f}% of all incidents"
        )
        
        # Analyze behavioral patterns
        if behavior_patterns.get('seasonal_concentration', 0) > 0.5:
            findings.append(
                "Strong seasonal concentration in wildlife activity patterns"
            )
            recommendations.append(
                "Implement species-specific management strategies aligned with "
                "seasonal patterns"
            )
        
        return InsightCategory(
            title="Species Patterns",
            findings=findings,
            recommendations=recommendations,
            priority="High",
            impact_areas=["Wildlife Management", "Risk Mitigation"],
            supporting_metrics={
                'top_species_concentration': top_species['incident_rate'].sum(),
                'seasonal_concentration': behavior_patterns.get('seasonal_concentration', 0)
            }
        )
    
    def analyze_operational_factors(
        self,
        operational_analysis: Dict
    ) -> InsightCategory:
        """
        Generate insights from operational analysis
        """
        findings = []
        recommendations = []
        
        # Analyze phase of flight patterns
        high_risk_phases = operational_analysis.get('high_risk_phases', [])
        if high_risk_phases:
            phases_str = ", ".join(high_risk_phases)
            findings.append(
                f"Higher risk during specific flight phases: {phases_str}"
            )
            recommendations.append(
                "Enhance monitoring and prevention measures during identified "
                "high-risk flight phases"
            )
        
        # Analyze effectiveness of current measures
        if operational_analysis.get('warning_effectiveness', 0) > 0.5:
            findings.append(
                "Current warning systems show significant effectiveness in "
                "reducing incidents"
            )
            recommendations.append(
                "Expand implementation of successful warning systems across "
                "more airports"
            )
        
        return InsightCategory(
            title="Operational Factors",
            findings=findings,
            recommendations=recommendations,
            priority="High" if high_risk_phases else "Medium",
            impact_areas=["Operations", "Safety Procedures", "Training"],
            supporting_metrics={
                'high_risk_phases_count': len(high_risk_phases),
                'warning_effectiveness': operational_analysis.get('warning_effectiveness', 0)
            }
        )
    
    def generate_executive_summary(self) -> str:
        """
        Generate executive summary from collected insights
        """
        summary = ["# Executive Summary\n"]
        
        # Add key findings
        summary.append("## Key Findings\n")
        for insight in self.insights:
            if insight.priority == "High":
                for finding in insight.findings:
                    summary.append(f"- {finding}")
        
        # Add priority recommendations
        summary.append("\n## Priority Recommendations\n")
        for insight in self.insights:
            if insight.priority == "High":
                for rec in insight.recommendations:
                    summary.append(f"- {rec}")
        
        return "\n".join(summary)
    
    def generate_detailed_report(self) -> str:
        """
        Generate detailed report with all insights
        """
        report = ["# Comprehensive Wildlife Strike Analysis Report\n"]
        
        # Add executive summary
        report.append(self.generate_executive_summary())
        
        # Add detailed sections
        for insight in self.insights:
            report.append(f"\n## {insight.title}\n")
            
            report.append("### Findings")
            for finding in insight.findings:
                report.append(f"- {finding}")
            
            report.append("\n### Recommendations")
            for rec in insight.recommendations:
                report.append(f"- {rec}")
            
            report.append("\n### Impact Areas")
            for area in insight.impact_areas:
                report.append(f"- {area}")
            
            report.append("\n### Supporting Metrics")
            for metric, value in insight.supporting_metrics.items():
                report.append(f"- {metric}: {value}")
        
        return "\n".join(report)
    
    def save_reports(self) -> None:
        """
        Save generated reports to files
        """
        # Save executive summary
        exec_summary = self.generate_executive_summary()
        with open(self.output_dir / "executive_summary.md", "w") as f:
            f.write(exec_summary)
        
        # Save detailed report
        detailed_report = self.generate_detailed_report()
        with open(self.output_dir / "detailed_report.md", "w") as f:
            f.write(detailed_report)
        
        # Save insights as structured data
        insights_data = []
        for insight in self.insights:
            insights_data.append({
                'title': insight.title,
                'findings': insight.findings,
                'recommendations': insight.recommendations,
                'priority': insight.priority,
                'impact_areas': insight.impact_areas,
                'supporting_metrics': insight.supporting_metrics
            })
        
        pd.DataFrame(insights_data).to_csv(
            self.output_dir / "insights_data.csv",
            index=False
        )