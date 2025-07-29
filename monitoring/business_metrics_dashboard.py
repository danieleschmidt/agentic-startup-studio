"""
Business Metrics Dashboard for Production Excellence
Real-time business KPI tracking, ROI analysis, and executive reporting system.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict, deque
import uuid
from pathlib import Path

# Dashboard generation (if available)
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import pandas as pd
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


class MetricType(Enum):
    """Types of business metrics."""
    REVENUE = "revenue"
    COST = "cost"
    EFFICIENCY = "efficiency"
    QUALITY = "quality"
    USER_ENGAGEMENT = "user_engagement"
    SYSTEM_PERFORMANCE = "system_performance"


class AlertSeverity(Enum):
    """Business alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    URGENT = "urgent"


@dataclass
class BusinessMetric:
    """Individual business metric data point."""
    metric_id: str
    name: str
    value: float
    unit: str
    category: MetricType
    timestamp: datetime
    target_value: Optional[float] = None
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
    
    @property
    def is_on_target(self) -> bool:
        """Check if metric meets target."""
        if self.target_value is None:
            return True
        return self.value >= self.target_value
    
    @property
    def alert_level(self) -> AlertSeverity:
        """Determine alert level based on thresholds."""
        if self.threshold_critical and self.value <= self.threshold_critical:
            return AlertSeverity.CRITICAL
        elif self.threshold_warning and self.value <= self.threshold_warning:
            return AlertSeverity.WARNING
        return AlertSeverity.INFO
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'metric_id': self.metric_id,
            'name': self.name,
            'value': self.value,
            'unit': self.unit,
            'category': self.category.value,
            'timestamp': self.timestamp.isoformat(),
            'target_value': self.target_value,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'is_on_target': self.is_on_target,
            'alert_level': self.alert_level.value,
            'metadata': self.metadata
        }


@dataclass
class KPISummary:
    """Key Performance Indicator summary."""
    period: str
    metrics: Dict[str, float]
    trends: Dict[str, str]  # "up", "down", "stable"
    alerts: List[str]
    roi_analysis: Dict[str, float]
    recommendations: List[str]


class BusinessMetricsCollector:
    """Collects and aggregates business metrics from various sources."""
    
    def __init__(self):
        self.metrics_history: deque = deque(maxlen=10000)
        self.real_time_metrics: Dict[str, BusinessMetric] = {}
        self.logger = logging.getLogger(f"{__name__}.BusinessMetricsCollector")
        
        # Business targets and thresholds
        self.targets = {
            'monthly_revenue': 50000.0,
            'user_acquisition_cost': 25.0,
            'customer_lifetime_value': 500.0,
            'system_uptime': 99.9,
            'api_response_time': 200.0,
            'user_satisfaction': 4.5,
            'feature_adoption_rate': 0.75,
            'churn_rate': 0.05
        }
        
        # Initialize baseline metrics
        self._initialize_baseline_metrics()
    
    def _initialize_baseline_metrics(self):
        """Initialize baseline business metrics."""
        baseline_metrics = [
            BusinessMetric(
                metric_id="monthly_revenue",
                name="Monthly Recurring Revenue",
                value=45000.0,
                unit="USD",
                category=MetricType.REVENUE,
                timestamp=datetime.now(),
                target_value=50000.0,
                threshold_warning=40000.0,
                threshold_critical=30000.0,
                metadata={"source": "billing_system", "confidence": 0.95}
            ),
            BusinessMetric(
                metric_id="user_acquisition_cost",
                name="Customer Acquisition Cost",
                value=28.50,
                unit="USD",
                category=MetricType.COST,
                timestamp=datetime.now(),
                target_value=25.0,
                threshold_warning=30.0,
                threshold_critical=40.0,
                metadata={"channel": "mixed", "period": "30_days"}
            ),
            BusinessMetric(
                metric_id="system_uptime",
                name="System Uptime Percentage",
                value=99.94,
                unit="%",
                category=MetricType.SYSTEM_PERFORMANCE,
                timestamp=datetime.now(),
                target_value=99.9,
                threshold_warning=99.0,
                threshold_critical=98.0,
                metadata={"monitoring_window": "24h", "service_count": 12}
            ),
            BusinessMetric(
                metric_id="api_response_time",
                name="API Average Response Time",
                value=185.0,
                unit="ms",
                category=MetricType.SYSTEM_PERFORMANCE,
                timestamp=datetime.now(),
                target_value=200.0,
                threshold_warning=300.0,
                threshold_critical=500.0,
                metadata={"percentile": "p95", "endpoints": 24}
            ),
            BusinessMetric(
                metric_id="user_satisfaction",
                name="User Satisfaction Score",
                value=4.6,
                unit="rating",
                category=MetricType.QUALITY,
                timestamp=datetime.now(),
                target_value=4.5,
                threshold_warning=4.0,
                threshold_critical=3.5,
                metadata={"survey_responses": 1247, "rating_scale": "1-5"}
            ),
            BusinessMetric(
                metric_id="feature_adoption_rate",
                name="New Feature Adoption Rate",
                value=0.78,
                unit="ratio",
                category=MetricType.USER_ENGAGEMENT,
                timestamp=datetime.now(),
                target_value=0.75,
                threshold_warning=0.60,
                threshold_critical=0.40,
                metadata={"feature": "ai_assistant", "user_base": 15000}
            )
        ]
        
        for metric in baseline_metrics:
            self.record_metric(metric)
    
    def record_metric(self, metric: BusinessMetric):
        """Record a new business metric."""
        self.metrics_history.append(metric)
        self.real_time_metrics[metric.metric_id] = metric
        
        self.logger.debug(f"Recorded metric: {metric.name} = {metric.value} {metric.unit}")
        
        # Log alerts for critical metrics
        if metric.alert_level in [AlertSeverity.CRITICAL, AlertSeverity.URGENT]:
            self.logger.warning(f"BUSINESS ALERT: {metric.name} is {metric.alert_level.value} - Value: {metric.value} {metric.unit}")
    
    def get_metrics_by_category(self, category: MetricType, hours: int = 24) -> List[BusinessMetric]:
        """Get metrics by category within time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [m for m in self.metrics_history 
                if m.category == category and m.timestamp >= cutoff_time]
    
    def get_metric_trend(self, metric_id: str, hours: int = 24) -> Tuple[str, float]:
        """Calculate trend for a specific metric."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        metric_values = [m.value for m in self.metrics_history 
                        if m.metric_id == metric_id and m.timestamp >= cutoff_time]
        
        if len(metric_values) < 2:
            return "stable", 0.0
        
        # Simple trend calculation
        first_half = metric_values[:len(metric_values)//2]
        second_half = metric_values[len(metric_values)//2:]
        
        avg_first = sum(first_half) / len(first_half)
        avg_second = sum(second_half) / len(second_half)
        
        change_percent = ((avg_second - avg_first) / avg_first) * 100 if avg_first != 0 else 0
        
        if change_percent > 5:
            return "up", change_percent
        elif change_percent < -5:
            return "down", change_percent
        else:
            return "stable", change_percent
    
    def calculate_roi_metrics(self) -> Dict[str, float]:
        """Calculate return on investment metrics."""
        current_revenue = self.real_time_metrics.get('monthly_revenue', {}).value if 'monthly_revenue' in self.real_time_metrics else 0
        acquisition_cost = self.real_time_metrics.get('user_acquisition_cost', {}).value if 'user_acquisition_cost' in self.real_time_metrics else 0
        
        # Simulated additional metrics for ROI calculation
        development_cost = 75000.0  # Monthly development costs
        infrastructure_cost = 12000.0  # Monthly infrastructure costs
        
        total_costs = development_cost + infrastructure_cost
        gross_profit = current_revenue - total_costs
        roi_percentage = (gross_profit / total_costs) * 100 if total_costs > 0 else 0
        
        return {
            'gross_profit': gross_profit,
            'total_costs': total_costs,
            'roi_percentage': roi_percentage,
            'revenue_per_employee': current_revenue / 25,  # Assuming 25 employees
            'cost_per_user': total_costs / 15000,  # Assuming 15k users
            'profit_margin': (gross_profit / current_revenue) * 100 if current_revenue > 0 else 0
        }


class BusinessDashboard:
    """Business metrics dashboard generator and manager."""
    
    def __init__(self, metrics_collector: BusinessMetricsCollector):
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(f"{__name__}.BusinessDashboard")
        self.dashboard_cache = {}
        self.last_generated = None
    
    def generate_executive_summary(self) -> KPISummary:
        """Generate executive summary with key insights."""
        current_time = datetime.now()
        
        # Key metrics
        key_metrics = {}
        trends = {}
        alerts = []
        
        for metric_id in ['monthly_revenue', 'user_acquisition_cost', 'system_uptime', 'user_satisfaction']:
            if metric_id in self.metrics_collector.real_time_metrics:
                metric = self.metrics_collector.real_time_metrics[metric_id]
                key_metrics[metric_id] = metric.value
                
                trend, change = self.metrics_collector.get_metric_trend(metric_id)
                trends[metric_id] = f"{trend} ({change:+.1f}%)"
                
                if metric.alert_level in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]:
                    alerts.append(f"{metric.name}: {metric.alert_level.value.upper()}")
        
        # ROI analysis
        roi_analysis = self.metrics_collector.calculate_roi_metrics()
        
        # Generate recommendations
        recommendations = self._generate_recommendations(key_metrics, trends, roi_analysis)
        
        return KPISummary(
            period=f"Current ({current_time.strftime('%Y-%m-%d %H:%M')})",
            metrics=key_metrics,
            trends=trends,
            alerts=alerts,
            roi_analysis=roi_analysis,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, metrics: Dict[str, float], trends: Dict[str, str], roi: Dict[str, float]) -> List[str]:
        """Generate business recommendations based on metrics."""
        recommendations = []
        
        # Revenue recommendations
        if 'monthly_revenue' in metrics and metrics['monthly_revenue'] < 50000:
            recommendations.append("Revenue below target - consider increasing sales efforts or pricing optimization")
        
        # Cost optimization
        if 'user_acquisition_cost' in metrics and metrics['user_acquisition_cost'] > 30:
            recommendations.append("High customer acquisition cost - optimize marketing channels and conversion funnel")
        
        # Performance recommendations
        if 'system_uptime' in metrics and metrics['system_uptime'] < 99.5:
            recommendations.append("System uptime below target - investigate infrastructure reliability")
        
        # ROI recommendations
        if roi['roi_percentage'] < 20:
            recommendations.append("ROI below 20% - focus on cost reduction or revenue growth initiatives")
        
        if roi['profit_margin'] < 15:
            recommendations.append("Low profit margin - evaluate pricing strategy and operational efficiency")
        
        # Growth recommendations
        for metric_id, trend in trends.items():
            if "down" in trend and metric_id in ['monthly_revenue', 'user_satisfaction']:
                recommendations.append(f"Declining {metric_id.replace('_', ' ')} - implement improvement strategy")
        
        return recommendations
    
    def generate_html_dashboard(self) -> str:
        """Generate HTML dashboard with business metrics."""
        if not PLOTLY_AVAILABLE:
            return self._generate_simple_html_dashboard()
        
        summary = self.generate_executive_summary()
        
        # Create visualizations
        dashboard_html = self._create_plotly_dashboard(summary)
        
        self.dashboard_cache['html'] = dashboard_html
        self.last_generated = datetime.now()
        
        return dashboard_html
    
    def _create_plotly_dashboard(self, summary: KPISummary) -> str:
        """Create interactive Plotly dashboard."""
        # Create subplot layout
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Revenue Metrics', 'System Performance', 'User Engagement', 'Cost Analysis', 'ROI Overview', 'Trends'),
            specs=[[{"type": "indicator"}, {"type": "indicator"}],
                   [{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "pie"}, {"type": "scatter"}]]
        )
        
        # Revenue indicator
        revenue_value = summary.metrics.get('monthly_revenue', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=revenue_value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Monthly Revenue (USD)"},
            delta={'reference': 50000},
            gauge={
                'axis': {'range': [None, 80000]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30000], 'color': "lightgray"},
                    {'range': [30000, 50000], 'color': "yellow"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50000}
            }
        ), row=1, col=1)
        
        # System uptime indicator
        uptime_value = summary.metrics.get('system_uptime', 0)
        fig.add_trace(go.Indicator(
            mode="gauge+number",
            value=uptime_value,
            title={'text': "System Uptime (%)"},
            gauge={
                'axis': {'range': [95, 100]},
                'bar': {'color': "green"},
                'steps': [{'range': [95, 99], 'color': "yellow"}],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 99.9}
            }
        ), row=1, col=2)
        
        # ROI bar chart
        roi_data = summary.roi_analysis
        fig.add_trace(go.Bar(
            x=['Revenue', 'Costs', 'Profit'],
            y=[summary.metrics.get('monthly_revenue', 0), roi_data['total_costs'], roi_data['gross_profit']],
            name='Financial Overview',
            marker_color=['green', 'red', 'blue']
        ), row=2, col=1)
        
        # Metric trends scatter plot
        trend_metrics = list(summary.metrics.keys())[:4]
        trend_values = [summary.metrics[m] for m in trend_metrics]
        
        fig.add_trace(go.Scatter(
            x=trend_metrics,
            y=trend_values,
            mode='lines+markers',
            name='Key Metrics',
            line=dict(color='purple', width=3)
        ), row=2, col=2)
        
        # Cost breakdown pie chart
        costs = {
            'Development': 75000,
            'Infrastructure': 12000,
            'Marketing': summary.metrics.get('user_acquisition_cost', 0) * 100,  # Estimated
            'Operations': 8000
        }
        
        fig.add_trace(go.Pie(
            labels=list(costs.keys()),
            values=list(costs.values()),
            name="Cost Breakdown"
        ), row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f"Business Metrics Dashboard - {summary.period}",
            height=1000,
            showlegend=True,
            template="plotly_white"
        )
        
        # Convert to HTML
        dashboard_html = fig.to_html(
            full_html=True,
            include_plotlyjs='cdn',
            div_id="business-dashboard",
            config={'displayModeBar': True, 'responsive': True}
        )
        
        # Add custom styling and additional info
        dashboard_html = self._add_dashboard_styling(dashboard_html, summary)
        
        return dashboard_html
    
    def _add_dashboard_styling(self, base_html: str, summary: KPISummary) -> str:
        """Add custom styling and information to dashboard."""
        
        alerts_html = ""
        if summary.alerts:
            alerts_html = "<div class='alerts'><h3>⚠️ Active Alerts</h3><ul>"
            for alert in summary.alerts:
                alerts_html += f"<li class='alert'>{alert}</li>"
            alerts_html += "</ul></div>"
        
        recommendations_html = "<div class='recommendations'><h3>💡 Recommendations</h3><ul>"
        for rec in summary.recommendations:
            recommendations_html += f"<li class='recommendation'>{rec}</li>"
        recommendations_html += "</ul></div>"
        
        roi_summary = f"""
        <div class='roi-summary'>
            <h3>📊 ROI Analysis</h3>
            <p><strong>ROI:</strong> {summary.roi_analysis['roi_percentage']:.1f}%</p>
            <p><strong>Profit Margin:</strong> {summary.roi_analysis['profit_margin']:.1f}%</p>
            <p><strong>Revenue per Employee:</strong> ${summary.roi_analysis['revenue_per_employee']:,.0f}</p>
        </div>
        """
        
        custom_content = f"""
        <style>
            body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
            .dashboard-header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .dashboard-header h1 {{ margin: 0; font-size: 2.5em; }}
            .dashboard-header p {{ margin: 5px 0; opacity: 0.9; }}
            .alerts {{ background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 15px; margin: 20px 0; }}
            .alert {{ color: #856404; margin: 5px 0; }}
            .recommendations {{ background: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 15px; margin: 20px 0; }}
            .recommendation {{ color: #0c5460; margin: 5px 0; }}
            .roi-summary {{ background: #e2e3e5; border: 1px solid #d6d8db; border-radius: 5px; padding: 15px; margin: 20px 0; }}
            .roi-summary p {{ margin: 8px 0; font-size: 1.1em; }}
            .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 0.9em; }}
        </style>
        <div class='dashboard-header'>
            <h1>🚀 Business Intelligence Dashboard</h1>
            <p>Real-time business metrics and performance analytics</p>
            <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
        </div>
        {alerts_html}
        {roi_summary}
        """
        
        # Insert custom content after <body> tag
        enhanced_html = base_html.replace('<body>', f'<body>{custom_content}')
        
        # Add recommendations and footer before </body>
        enhanced_html = enhanced_html.replace('</body>', f'{recommendations_html}<div class="footer">Generated by Terragon Business Intelligence System</div></body>')
        
        return enhanced_html
    
    def _generate_simple_html_dashboard(self) -> str:
        """Generate simple HTML dashboard without Plotly."""
        summary = self.generate_executive_summary()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Business Metrics Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f8f9fa; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; text-align: center; }}
                .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 30px 0; }}
                .metric-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .metric-value {{ font-size: 2.5em; font-weight: bold; color: #2c3e50; }}
                .metric-name {{ color: #7f8c8d; font-size: 1.1em; margin-bottom: 10px; }}
                .metric-trend {{ font-size: 0.9em; margin-top: 10px; }}
                .trend-up {{ color: #27ae60; }}
                .trend-down {{ color: #e74c3c; }}
                .trend-stable {{ color: #f39c12; }}
                .alerts {{ background: #fff3cd; border-left: 4px solid #ffc107; padding: 20px; margin: 20px 0; }}
                .recommendations {{ background: #d1ecf1; border-left: 4px solid #17a2b8; padding: 20px; margin: 20px 0; }}
                .roi-section {{ background: white; border-radius: 8px; padding: 25px; margin: 20px 0; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .roi-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; margin-top: 15px; }}
                .roi-item {{ text-align: center; padding: 15px; background: #f8f9fa; border-radius: 5px; }}
                .roi-value {{ font-size: 1.8em; font-weight: bold; color: #495057; }}
                .roi-label {{ color: #6c757d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Business Intelligence Dashboard</h1>
                <p>Real-time Business Metrics & Performance Analytics</p>
                <p>{summary.period}</p>
            </div>
            
            <div class="metrics-grid">
        """
        
        # Add metric cards
        for metric_id, value in summary.metrics.items():
            trend_info = summary.trends.get(metric_id, "stable (0.0%)")
            trend_class = "trend-up" if "up" in trend_info else "trend-down" if "down" in trend_info else "trend-stable"
            
            metric_name = metric_id.replace('_', ' ').title()
            unit = self._get_metric_unit(metric_id)
            
            html += f"""
                <div class="metric-card">
                    <div class="metric-name">{metric_name}</div>
                    <div class="metric-value">{value:,.1f}{unit}</div>
                    <div class="metric-trend {trend_class}">📈 {trend_info}</div>
                </div>
            """
        
        html += "</div>"
        
        # Add alerts section
        if summary.alerts:
            html += """
            <div class="alerts">
                <h3>⚠️ Active Alerts</h3>
                <ul>
            """
            for alert in summary.alerts:
                html += f"<li>{alert}</li>"
            html += "</ul></div>"
        
        # Add ROI section
        html += f"""
        <div class="roi-section">
            <h3>📊 ROI Analysis</h3>
            <div class="roi-grid">
                <div class="roi-item">
                    <div class="roi-value">{summary.roi_analysis['roi_percentage']:.1f}%</div>
                    <div class="roi-label">Return on Investment</div>
                </div>
                <div class="roi-item">
                    <div class="roi-value">${summary.roi_analysis['gross_profit']:,.0f}</div>
                    <div class="roi-label">Gross Profit</div>
                </div>
                <div class="roi-item">
                    <div class="roi-value">{summary.roi_analysis['profit_margin']:.1f}%</div>
                    <div class="roi-label">Profit Margin</div>
                </div>
                <div class="roi-item">
                    <div class="roi-value">${summary.roi_analysis['revenue_per_employee']:,.0f}</div>
                    <div class="roi-label">Revenue per Employee</div>
                </div>
            </div>
        </div>
        """
        
        # Add recommendations
        html += """
        <div class="recommendations">
            <h3>💡 Strategic Recommendations</h3>
            <ul>
        """
        for rec in summary.recommendations:
            html += f"<li>{rec}</li>"
        
        html += f"""
            </ul>
        </div>
        
        <div style="text-align: center; margin-top: 40px; color: #6c757d; font-size: 0.9em;">
            Generated by Terragon Business Intelligence System • {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        </div>
        
        </body>
        </html>
        """
        
        return html
    
    def _get_metric_unit(self, metric_id: str) -> str:
        """Get display unit for metric."""
        units = {
            'monthly_revenue': ' USD',
            'user_acquisition_cost': ' USD',
            'system_uptime': '%',
            'api_response_time': 'ms',
            'user_satisfaction': '/5',
            'feature_adoption_rate': '%'
        }
        return units.get(metric_id, '')
    
    def save_dashboard(self, output_path: str = "business_dashboard.html"):
        """Save dashboard to HTML file."""
        dashboard_html = self.generate_html_dashboard()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(dashboard_html)
        
        self.logger.info(f"Business dashboard saved to {output_path}")
        return output_path


# Standalone dashboard validation
async def validate_business_dashboard():
    """Validate business dashboard functionality."""
    print("📊 Business Metrics Dashboard Validation")
    print("=" * 50)
    
    # Initialize metrics collector
    metrics_collector = BusinessMetricsCollector()
    
    print("\n1. Testing Metrics Collection")
    # Add some dynamic metrics
    test_metrics = [
        BusinessMetric(
            metric_id="daily_active_users",
            name="Daily Active Users",
            value=12500.0,
            unit="users",
            category=MetricType.USER_ENGAGEMENT,
            timestamp=datetime.now(),
            target_value=15000.0,
            metadata={"growth_rate": 0.05}
        ),
        BusinessMetric(
            metric_id="conversion_rate",
            name="Trial to Paid Conversion Rate",
            value=0.23,
            unit="ratio",
            category=MetricType.EFFICIENCY,
            timestamp=datetime.now(),
            target_value=0.25,
            threshold_warning=0.20,
            threshold_critical=0.15
        )
    ]
    
    for metric in test_metrics:
        metrics_collector.record_metric(metric)
    
    print(f"   ✅ Recorded {len(test_metrics)} additional metrics")
    print(f"   📈 Total metrics in system: {len(metrics_collector.metrics_history)}")
    
    print("\n2. Testing Dashboard Generation")
    dashboard = BusinessDashboard(metrics_collector)
    
    # Generate executive summary
    summary = dashboard.generate_executive_summary()
    print(f"   📋 Generated executive summary for period: {summary.period}")
    print(f"   📊 Key metrics tracked: {len(summary.metrics)}")
    print(f"   ⚠️  Active alerts: {len(summary.alerts)}")
    print(f"   💡 Recommendations: {len(summary.recommendations)}")
    
    print("\n3. Key Performance Indicators")
    for metric_id, value in summary.metrics.items():
        trend = summary.trends.get(metric_id, "stable")
        metric_name = metric_id.replace('_', ' ').title()
        print(f"   {metric_name}: {value:.1f} ({trend})")
    
    print("\n4. ROI Analysis")
    roi = summary.roi_analysis
    print(f"   ROI Percentage: {roi['roi_percentage']:.1f}%")
    print(f"   Profit Margin: {roi['profit_margin']:.1f}%")
    print(f"   Revenue per Employee: ${roi['revenue_per_employee']:,.0f}")
    print(f"   Gross Profit: ${roi['gross_profit']:,.0f}")
    
    print("\n5. Business Alerts")
    if summary.alerts:
        for alert in summary.alerts:
            print(f"   🚨 {alert}")
    else:
        print("   ✅ No critical alerts")
    
    print("\n6. Strategic Recommendations")
    for i, rec in enumerate(summary.recommendations, 1):
        print(f"   {i}. {rec}")
    
    print("\n7. Dashboard File Generation")
    dashboard_path = dashboard.save_dashboard("test_business_dashboard.html")
    print(f"   💾 Dashboard saved to: {dashboard_path}")
    
    # Validate dashboard file exists and has content
    dashboard_file = Path(dashboard_path)
    if dashboard_file.exists():
        file_size = dashboard_file.stat().st_size
        print(f"   📏 Dashboard file size: {file_size:,} bytes")
        
        if file_size > 5000:  # Reasonable size for HTML dashboard
            print("   ✅ Dashboard file generated successfully")
            validation_passed = True
        else:
            print("   ❌ Dashboard file too small - may be incomplete")
            validation_passed = False
    else:
        print("   ❌ Dashboard file not created")
        validation_passed = False
    
    print("\n8. Performance Metrics")
    healthy_metrics = sum(1 for m in metrics_collector.real_time_metrics.values() if m.is_on_target)
    total_metrics = len(metrics_collector.real_time_metrics)
    health_percentage = (healthy_metrics / total_metrics) * 100 if total_metrics > 0 else 0
    
    print(f"   🎯 Metrics on target: {healthy_metrics}/{total_metrics} ({health_percentage:.1f}%)")
    print(f"   📈 ROI above 15%: {'Yes' if roi['roi_percentage'] > 15 else 'No'}")
    print(f"   💰 Revenue target met: {'Yes' if summary.metrics.get('monthly_revenue', 0) >= 50000 else 'No'}")
    
    print(f"\n✅ Business Dashboard Validation Complete")
    print(f"🎯 Overall Health Score: {health_percentage:.1f}%")
    print(f"📊 Dashboard Generated: {'Yes' if validation_passed else 'No'}")
    
    return validation_passed and health_percentage >= 60  # 60% of metrics should be on target


if __name__ == "__main__":
    # Run standalone validation
    compliance = asyncio.run(validate_business_dashboard())
    print(f"\n🏆 Business Dashboard Target: {'✅ ACHIEVED' if compliance else '❌ NOT MET'}")
    exit(0 if compliance else 1)