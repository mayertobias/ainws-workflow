"""
Professional Report Generator for Music Analysis

Creates comprehensive PDF and HTML reports with:
- Visual charts and graphs
- Industry-standard formatting
- Executive summaries
- Actionable recommendations
- Export functionality
"""

import logging
import json
import io
import base64
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# For PDF generation
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.charts.piecharts import Pie
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    logging.warning("ReportLab not available. PDF generation will be disabled.")

logger = logging.getLogger(__name__)

class ProfessionalReportGenerator:
    """
    Professional report generator for music industry analysis.
    
    Creates publication-ready reports with:
    - Executive summaries
    - Visual analytics
    - Actionable recommendations
    - Industry benchmarking
    - Export capabilities
    """
    
    def __init__(self, output_dir: str = "./reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        if PDF_AVAILABLE:
            self.styles = getSampleStyleSheet()
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for professional reports"""
        if not PDF_AVAILABLE:
            return
            
        # Executive Summary Style
        self.styles.add(ParagraphStyle(
            name='ExecutiveSummary',
            parent=self.styles['Normal'],
            fontSize=11,
            leading=14,
            spaceAfter=12,
            leftIndent=20,
            rightIndent=20,
            backColor=colors.lightgrey,
            borderPadding=10
        ))
        
        # Recommendation Style
        self.styles.add(ParagraphStyle(
            name='Recommendation',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=8,
            leftIndent=30,
            bulletIndent=10,
            bulletText='•'
        ))
        
        # Section Header Style
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            leading=18,
            spaceAfter=12,
            textColor=colors.darkblue,
            borderWidth=1,
            borderColor=colors.darkblue,
            borderPadding=5
        ))
        
        # Key Insight Style
        self.styles.add(ParagraphStyle(
            name='KeyInsight',
            parent=self.styles['Normal'],
            fontSize=10,
            leading=12,
            spaceAfter=6,
            leftIndent=15,
            backColor=colors.lightblue,
            borderPadding=5
        ))
    
    def generate_comprehensive_report(self, 
                                    analysis_results: Dict[str, Any],
                                    song_metadata: Dict[str, Any],
                                    export_formats: List[str] = ['html', 'pdf']) -> Dict[str, str]:
        """
        Generate comprehensive analysis report in multiple formats
        
        Args:
            analysis_results: Results from enhanced analysis service
            song_metadata: Song information and metadata
            export_formats: List of formats to generate ['html', 'pdf', 'json']
            
        Returns:
            Dictionary with file paths for each generated format
        """
        try:
            # Create visualizations
            charts = self._create_analysis_charts(analysis_results, song_metadata)
            
            # Generate report data structure
            report_data = self._structure_report_data(analysis_results, song_metadata, charts)
            
            # Generate reports in requested formats
            generated_files = {}
            
            if 'html' in export_formats:
                html_path = self._generate_html_report(report_data)
                generated_files['html'] = html_path
            
            if 'pdf' in export_formats and PDF_AVAILABLE:
                pdf_path = self._generate_pdf_report(report_data)
                generated_files['pdf'] = pdf_path
            
            if 'json' in export_formats:
                json_path = self._generate_json_export(analysis_results, song_metadata)
                generated_files['json'] = json_path
            
            return generated_files
            
        except Exception as e:
            logger.error(f"Error generating comprehensive report: {e}")
            raise
    
    def _create_analysis_charts(self, analysis_results: Dict[str, Any], song_metadata: Dict[str, Any]) -> Dict[str, str]:
        """Create visual charts for the analysis with fallback support"""
        charts = {}
        
        try:
            logger.info("Creating analysis charts...")
            
            # 1. Feature Radar Chart - with fallback
            try:
                if 'hit_comparison' in analysis_results:
                    charts['feature_radar'] = self._create_feature_radar_chart(analysis_results['hit_comparison'])
                else:
                    # Create fallback radar chart from available data
                    charts['feature_radar'] = self._create_fallback_radar_chart(analysis_results)
            except Exception as e:
                logger.warning(f"Feature radar chart failed: {e}")
                charts['feature_radar'] = self._create_simple_chart("Feature Analysis", "Charts unavailable")
            
            # 2. Hit Probability Chart - with fallback
            try:
                if 'hit_comparison' in analysis_results:
                    charts['hit_probability'] = self._create_hit_probability_chart(analysis_results['hit_comparison'])
                else:
                    # Create simple probability chart from overall assessment
                    overall_score = analysis_results.get('overall_assessment', {}).get('overall_score', 0.5)
                    charts['hit_probability'] = self._create_simple_probability_chart(overall_score)
            except Exception as e:
                logger.warning(f"Hit probability chart failed: {e}")
                charts['hit_probability'] = self._create_simple_chart("Hit Probability", "Charts unavailable")
            
            # 3. Innovation Scores Chart - with fallback  
            try:
                if 'novelty_assessment' in analysis_results:
                    charts['innovation_scores'] = self._create_innovation_chart(analysis_results['novelty_assessment'])
                else:
                    charts['innovation_scores'] = self._create_simple_chart("Innovation Analysis", "Innovation assessment unavailable")
            except Exception as e:
                logger.warning(f"Innovation chart failed: {e}")
                charts['innovation_scores'] = self._create_simple_chart("Innovation Analysis", "Charts unavailable")
            
            # 4. Production Quality Dashboard - with fallback
            try:
                if 'production_analysis' in analysis_results:
                    charts['production_quality'] = self._create_production_dashboard(analysis_results['production_analysis'])
                else:
                    charts['production_quality'] = self._create_simple_chart("Production Quality", "Production analysis unavailable")
            except Exception as e:
                logger.warning(f"Production dashboard failed: {e}")
                charts['production_quality'] = self._create_simple_chart("Production Quality", "Charts unavailable")
            
            logger.info(f"Charts created successfully: {list(charts.keys())}")
            
        except Exception as e:
            logger.error(f"Error creating charts: {e}")
            # Ensure we have at least basic charts
            charts = {
                'feature_radar': self._create_simple_chart("Feature Analysis", "Chart generation failed"),
                'hit_probability': self._create_simple_chart("Hit Probability", "Chart generation failed"),
                'innovation_scores': self._create_simple_chart("Innovation Analysis", "Chart generation failed"),
                'production_quality': self._create_simple_chart("Production Quality", "Chart generation failed")
            }
        
        return charts
    
    def _create_feature_radar_chart(self, hit_comparison: Dict[str, Any]) -> str:
        """Create radar chart showing feature alignment with genre norms"""
        try:
            # Extract feature analysis data
            feature_analysis = hit_comparison.get('feature_analysis', {})
            aligned_features = feature_analysis.get('aligned_features', {})
            
            if not aligned_features:
                return ""
            
            # Prepare data for radar chart
            features = list(aligned_features.keys())
            values = []
            
            for feature, assessment in aligned_features.items():
                # Convert assessment to numerical score (0-1)
                if isinstance(assessment, str):
                    if 'perfectly aligned' in assessment.lower():
                        values.append(1.0)
                    elif 'slightly' in assessment.lower():
                        values.append(0.8)
                    elif 'notably' in assessment.lower():
                        values.append(0.6)
                    else:
                        values.append(0.4)
                else:
                    values.append(0.5)
            
            # Create radar chart
            fig = go.Figure(data=go.Scatterpolar(
                r=values,
                theta=features,
                fill='toself',
                name='Feature Alignment',
                line=dict(color='rgb(46, 125, 50)', width=3),
                fillcolor='rgba(46, 125, 50, 0.3)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1],
                        tickvals=[0, 0.5, 1],
                        ticktext=['Poor', 'Average', 'Excellent']
                    ),
                    angularaxis=dict(
                        direction='clockwise',
                        tickfont_size=12
                    )
                ),
                title=dict(
                    text='Feature Alignment with Genre Norms',
                    x=0.5,
                    font=dict(size=16, color='rgb(37, 37, 37)')
                ),
                showlegend=False,
                width=500,
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            # Convert to base64 for embedding
            img_bytes = fig.to_image(format="png", width=500, height=500, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating radar chart: {e}")
            return ""
    
    def _create_hit_probability_chart(self, hit_comparison: Dict[str, Any]) -> str:
        """Create hit probability visualization"""
        try:
            statistical_analysis = hit_comparison.get('statistical_analysis', {})
            hit_probability = statistical_analysis.get('overall_hit_probability', '50%')
            
            # Extract percentage value
            prob_value = float(hit_probability.replace('%', '')) if '%' in hit_probability else 50.0
            
            # Create gauge chart
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = prob_value,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Hit Potential Score"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(
                width=400,
                height=300,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            img_bytes = fig.to_image(format="png", width=400, height=300, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating hit probability chart: {e}")
            return ""
    
    def _create_innovation_chart(self, novelty_assessment: Dict[str, Any]) -> str:
        """Create innovation scores bar chart"""
        try:
            innovation_scores = novelty_assessment.get('innovation_scores', {})
            
            if not innovation_scores:
                return ""
            
            # Prepare data
            categories = list(innovation_scores.keys())
            scores = [float(score) for score in innovation_scores.values()]
            
            # Create bar chart
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=scores,
                    marker_color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'],
                    text=[f'{score:.1f}' for score in scores],
                    textposition='auto',
                )
            ])
            
            fig.update_layout(
                title='Innovation Assessment Scores',
                xaxis_title='Innovation Categories',
                yaxis_title='Score (0-1)',
                yaxis=dict(range=[0, 1]),
                width=600,
                height=400,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            img_bytes = fig.to_image(format="png", width=600, height=400, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating innovation chart: {e}")
            return ""
    
    def _create_production_dashboard(self, production_analysis: Dict[str, Any]) -> str:
        """Create production quality dashboard"""
        try:
            technical_assessment = production_analysis.get('technical_assessment', {})
            mix_evaluation = production_analysis.get('mix_evaluation', {})
            
            # Create subplot dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Technical Quality', 'Mix Scores', 'Platform Readiness', 'Overall Grade'),
                specs=[[{"type": "indicator"}, {"type": "bar"}],
                       [{"type": "pie"}, {"type": "indicator"}]]
            )
            
            # Technical Quality Gauge
            grade = technical_assessment.get('overall_technical_grade', 'B')
            grade_value = {'A': 95, 'B': 85, 'C': 75, 'D': 65, 'F': 50}.get(grade, 75)
            
            fig.add_trace(go.Indicator(
                mode="gauge+number",
                value=grade_value,
                title={'text': "Technical Grade"},
                gauge={'axis': {'range': [0, 100]},
                       'bar': {'color': "darkgreen"},
                       'steps': [{'range': [0, 60], 'color': "lightgray"},
                                {'range': [60, 80], 'color': "yellow"},
                                {'range': [80, 100], 'color': "lightgreen"}]},
                domain={'x': [0, 1], 'y': [0, 1]}
            ), row=1, col=1)
            
            # Mix Scores Bar Chart
            mix_scores = ['Clarity', 'Depth', 'Balance']
            mix_values = [
                float(mix_evaluation.get('clarity_score', '7/10').split('/')[0]),
                float(mix_evaluation.get('depth_score', '7/10').split('/')[0]),
                float(mix_evaluation.get('balance_score', '7/10').split('/')[0])
            ]
            
            fig.add_trace(go.Bar(
                x=mix_scores,
                y=mix_values,
                marker_color='rgb(55, 83, 109)'
            ), row=1, col=2)
            
            fig.update_layout(
                width=800,
                height=600,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)'
            )
            
            img_bytes = fig.to_image(format="png", width=800, height=600, scale=2)
            img_base64 = base64.b64encode(img_bytes).decode('utf-8')
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Error creating production dashboard: {e}")
            return ""
    
    def _structure_report_data(self, analysis_results: Dict[str, Any], 
                              song_metadata: Dict[str, Any], 
                              charts: Dict[str, str]) -> Dict[str, Any]:
        """Structure all data for report generation with robust fallbacks"""
        
        # Check if this is frontend AI insights format (new format)
        if 'executive_summary' in analysis_results and isinstance(analysis_results.get('executive_summary'), str):
            logger.info("Processing frontend AI insights format")
            return self._structure_frontend_insights_data(analysis_results, song_metadata, charts)
        
        # Original backend format handling
        logger.info("Processing backend orchestrator format")
        # Extract executive summary from comprehensive analysis with fallbacks
        comprehensive = analysis_results.get('comprehensive_analysis', {})
        executive_summary = comprehensive.get('executive_summary', {})
        
        # If no executive summary, create one from available data
        if not executive_summary:
            overall_assessment = analysis_results.get('overall_assessment', {})
            executive_summary = {
                'commercial_potential': overall_assessment.get('commercial_score', 0.5),
                'market_positioning': 'Analysis available in detailed report',
                'key_strengths': analysis_results.get('insights', ['Detailed analysis completed'])[:3],
                'improvement_areas': analysis_results.get('recommendations', ['Review detailed findings'])[:3]
            }
        
        # Extract action plan with fallbacks
        action_plan = comprehensive.get('action_plan', {})
        if not action_plan:
            action_plan = {
                'production_priorities': analysis_results.get('recommendations', ['No specific recommendations'])[:2],
                'arrangement_improvements': ['Review detailed analysis'],
                'lyrical_enhancements': ['Content analysis available']
            }
        
        # Extract market strategy with fallbacks
        market_strategy = comprehensive.get('market_strategy', {})
        if not market_strategy:
            market_strategy = {
                'target_demographics': 'General music audience',
                'marketing_channels': ['Streaming platforms', 'Social media'],
                'positioning_strategy': 'Quality music with commercial appeal'
            }
        
        return {
            'metadata': {
                'song_title': song_metadata.get('title', 'Unknown'),
                'artist': song_metadata.get('artist', 'Unknown'),
                'genre': song_metadata.get('genre', 'Unknown'),
                'analysis_date': datetime.now().strftime('%B %d, %Y'),
                'report_version': '2.0.0'
            },
            'executive_summary': executive_summary,
            'action_plan': action_plan,
            'market_strategy': market_strategy,
            'detailed_analysis': {
                'musical_meaning': analysis_results.get('musical_meaning', {}),
                'hit_comparison': analysis_results.get('hit_comparison', {}),
                'novelty_assessment': analysis_results.get('novelty_assessment', {}),
                'production_analysis': analysis_results.get('production_analysis', {})
            },
            'charts': charts,
            'recommendations': self._extract_all_recommendations(analysis_results)
        }
    
    def _extract_all_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract and categorize all recommendations"""
        recommendations = {
            'production': [],
            'arrangement': [],
            'lyrics': [],
            'marketing': [],
            'immediate_actions': [],
            'strategic': []
        }
        
        # Extract from comprehensive analysis
        comprehensive = analysis_results.get('comprehensive_analysis', {})
        action_plan = comprehensive.get('action_plan', {})
        
        recommendations['production'] = action_plan.get('production_priorities', [])
        recommendations['arrangement'] = action_plan.get('arrangement_improvements', [])
        recommendations['lyrics'] = action_plan.get('lyrical_enhancements', [])
        
        # Extract from production analysis
        production = analysis_results.get('production_analysis', {})
        improvement_plan = production.get('improvement_plan', {})
        
        recommendations['immediate_actions'].extend(improvement_plan.get('critical_fixes', []))
        recommendations['immediate_actions'].extend(improvement_plan.get('enhancement_suggestions', []))
        
        # Extract from hit comparison
        hit_comparison = analysis_results.get('hit_comparison', {})
        optimization = hit_comparison.get('optimization_strategy', {})
        
        recommendations['strategic'].extend(optimization.get('immediate_actions', []))
        
        return recommendations
    
    def _structure_frontend_insights_data(self, analysis_results: Dict[str, Any], 
                                        song_metadata: Dict[str, Any], 
                                        charts: Dict[str, str]) -> Dict[str, Any]:
        """Structure frontend AI insights data for report generation"""
        
        # Extract overall assessment scores
        overall_assessment = analysis_results.get('overall_assessment', {})
        commercial_score = overall_assessment.get('commercial_score', 75)
        artistic_score = overall_assessment.get('artistic_score', 75)
        innovation_score = overall_assessment.get('innovation_score', 60)
        
        # Convert scores to 0-1 range if they're in 0-100 range
        if commercial_score > 1:
            commercial_score = commercial_score / 100
        if artistic_score > 1:
            artistic_score = artistic_score / 100
        if innovation_score > 1:
            innovation_score = innovation_score / 100
        
        # Structure executive summary from frontend insights
        executive_summary = {
            'commercial_potential': commercial_score,
            'market_positioning': analysis_results.get('market_positioning', 'Professional music analysis completed'),
            'key_strengths': analysis_results.get('key_insights', {}).get('musical_insights', [])[:3],
            'improvement_areas': analysis_results.get('technical_improvements', [])[:3]
        }
        
        # Structure action plan from frontend insights
        action_plan = {
            'production_priorities': analysis_results.get('production_feedback', [])[:3],
            'arrangement_improvements': analysis_results.get('key_insights', {}).get('musical_insights', [])[:2],
            'lyrical_enhancements': analysis_results.get('key_insights', {}).get('commercial_insights', [])[:2]
        }
        
        # Structure market strategy from frontend insights
        market_strategy = {
            'target_demographics': analysis_results.get('target_demographics', ['General music audience'])[0] if analysis_results.get('target_demographics') else 'General music audience',
            'marketing_channels': ['Streaming platforms', 'Social media', 'Radio'],
            'positioning_strategy': analysis_results.get('competitive_analysis', 'Quality music with commercial appeal')
        }
        
        # Extract agent contributions for detailed analysis
        music_analysis = analysis_results.get('music_analysis', {})
        commercial_analysis = analysis_results.get('commercial_analysis', {})
        novelty_analysis = analysis_results.get('novelty_analysis', {})
        
        return {
            'metadata': {
                'song_title': song_metadata.get('title', 'Unknown'),
                'artist': song_metadata.get('artist', 'Unknown'),
                'genre': song_metadata.get('genre', 'Unknown'),
                'analysis_date': datetime.now().strftime('%B %d, %Y'),
                'report_version': '2.0.0',
                'architecture': analysis_results.get('architecture', 'enhanced_parallel_agentic_workflow'),
                'agents_used': analysis_results.get('agents_used', ['TuneScope', 'MarketMind', 'TrendScope'])
            },
            'executive_summary': executive_summary,
            'action_plan': action_plan,
            'market_strategy': market_strategy,
            'detailed_analysis': {
                'musical_meaning': {
                    'summary': analysis_results.get('executive_summary', ''),
                    'insights': music_analysis.get('key_insights', []),
                    'recommendations': music_analysis.get('recommendations', []),
                    'confidence_score': music_analysis.get('confidence_score', 0.8)
                },
                'hit_comparison': {
                    'overall_score': commercial_score,
                    'insights': commercial_analysis.get('key_insights', []),
                    'recommendations': commercial_analysis.get('recommendations', []),
                    'confidence_score': commercial_analysis.get('confidence_score', 0.8)
                },
                'novelty_assessment': novelty_analysis,
                'production_analysis': {
                    'feedback': analysis_results.get('production_feedback', []),
                    'strengths': analysis_results.get('technical_strengths', []),
                    'improvements': analysis_results.get('technical_improvements', [])
                }
            },
            'charts': charts,
            'recommendations': self._extract_frontend_recommendations(analysis_results),
            'frontend_insights': True  # Flag to indicate this is from frontend
        }
    
    def _extract_frontend_recommendations(self, analysis_results: Dict[str, Any]) -> Dict[str, List[str]]:
        """Extract and categorize recommendations from frontend AI insights"""
        recommendations = {
            'production': analysis_results.get('production_feedback', [])[:3],
            'arrangement': analysis_results.get('key_insights', {}).get('musical_insights', [])[:2],
            'lyrics': analysis_results.get('key_insights', {}).get('commercial_insights', [])[:2],
            'marketing': analysis_results.get('target_demographics', [])[:2],
            'immediate_actions': analysis_results.get('technical_improvements', [])[:3],
            'strategic': [analysis_results.get('competitive_analysis', 'Develop unique positioning strategy')]
        }
        
        # Add agent-specific recommendations
        if analysis_results.get('music_analysis', {}).get('recommendations'):
            recommendations['production'].extend(analysis_results['music_analysis']['recommendations'][:2])
        
        if analysis_results.get('commercial_analysis', {}).get('recommendations'):
            recommendations['marketing'].extend(analysis_results['commercial_analysis']['recommendations'][:2])
        
        return recommendations
    
    def _generate_html_report(self, report_data: Dict[str, Any]) -> str:
        """Generate professional HTML report"""
        try:
            metadata = report_data['metadata']
            executive_summary = report_data['executive_summary']
            charts = report_data['charts']
            recommendations = report_data['recommendations']
            
            html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Music Analysis Report - {metadata['song_title']}</title>
    <style>
        body {{
            font-family: 'Georgia', 'Times New Roman', serif;
            line-height: 1.6;
            color: #2c3e50;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            text-align: center;
            margin-bottom: 30px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .header h1 {{
            margin: 0;
            font-size: 2.5em;
            font-weight: bold;
        }}
        
        .header .subtitle {{
            font-size: 1.2em;
            opacity: 0.9;
            margin-top: 10px;
        }}
        
        .executive-summary {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 30px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 5px solid #3498db;
        }}
        
        .executive-summary h2 {{
            color: #2980b9;
            margin-top: 0;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 10px;
        }}
        
        .commercial-potential {{
            display: inline-block;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            color: white;
            margin: 5px 0;
        }}
        
        .commercial-potential.high {{
            background-color: #27ae60;
        }}
        
        .commercial-potential.medium {{
            background-color: #f39c12;
        }}
        
        .commercial-potential.low {{
            background-color: #e74c3c;
        }}
        
        .section {{
            background: white;
            padding: 25px;
            border-radius: 10px;
            margin-bottom: 25px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .section h2 {{
            color: #34495e;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
            margin-top: 0;
        }}
        
        .chart-container {{
            text-align: center;
            margin: 20px 0;
            padding: 15px;
            background: #f8f9fa;
            border-radius: 8px;
        }}
        
        .chart-container img {{
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        
        .recommendations {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 20px;
            margin-top: 20px;
        }}
        
        .recommendation-category {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border-left: 4px solid #3498db;
        }}
        
        .recommendation-category h3 {{
            margin-top: 0;
            color: #2980b9;
        }}
        
        .recommendation-category ul {{
            margin: 0;
            padding-left: 20px;
        }}
        
        .recommendation-category li {{
            margin-bottom: 8px;
            line-height: 1.4;
        }}
        
        .metadata-table {{
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }}
        
        .metadata-table th,
        .metadata-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        
        .metadata-table th {{
            background-color: #34495e;
            color: white;
        }}
        
        .key-insight {{
            background: linear-gradient(90deg, #74b9ff, #0984e3);
            color: white;
            padding: 15px;
            border-radius: 8px;
            margin: 15px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .key-insight strong {{
            display: block;
            font-size: 1.1em;
            margin-bottom: 5px;
        }}
        
        @media print {{
            body {{
                background-color: white;
            }}
            .section, .executive-summary {{
                box-shadow: none;
                border: 1px solid #ddd;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Professional Music Analysis Report</h1>
        <div class="subtitle">{metadata['song_title']} by {metadata['artist']}</div>
        <div class="subtitle">Analysis Date: {metadata['analysis_date']}</div>
    </div>
    
    <div class="executive-summary">
        <h2>Executive Summary</h2>
        <table class="metadata-table">
            <tr>
                <th>Song Title</th>
                <td>{metadata['song_title']}</td>
                <th>Artist</th>
                <td>{metadata['artist']}</td>
            </tr>
            <tr>
                <th>Genre</th>
                <td>{metadata['genre']}</td>
                <th>Analysis Version</th>
                <td>{metadata['report_version']}</td>
            </tr>
        </table>
        
        <div class="key-insight">
            <strong>Commercial Potential Assessment</strong>
            <span class="commercial-potential {self._format_commercial_potential_class(executive_summary.get('commercial_potential', 'medium'))}">
                {self._format_commercial_potential_text(executive_summary.get('commercial_potential', 'Medium'))} Potential
            </span>
            <br>
            <strong>Confidence Level:</strong> {self._safe_string_format(executive_summary.get('confidence_level', 'N/A'))}
        </div>
        
        <div class="key-insight">
            <strong>Investment Recommendation</strong>
            {self._safe_string_format(executive_summary.get('investment_recommendation', 'Analysis pending'))}
        </div>
    </div>
    
    <div class="section">
        <h2>Visual Analytics</h2>
        <div class="chart-container">
            <h3>Feature Alignment Analysis</h3>
            {f'<img src="{charts["feature_radar"]}" alt="Feature Radar Chart">' if charts.get('feature_radar') else '<p>Chart not available</p>'}
        </div>
        
        <div class="chart-container">
            <h3>Hit Potential Score</h3>
            {f'<img src="{charts["hit_probability"]}" alt="Hit Probability Chart">' if charts.get('hit_probability') else '<p>Chart not available</p>'}
        </div>
        
        <div class="chart-container">
            <h3>Innovation Assessment</h3>
            {f'<img src="{charts["innovation_scores"]}" alt="Innovation Scores Chart">' if charts.get('innovation_scores') else '<p>Chart not available</p>'}
        </div>
    </div>
    
    <div class="section">
        <h2>Strategic Recommendations</h2>
        <div class="recommendations">
            <div class="recommendation-category">
                <h3>🎚️ Production Priorities</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in recommendations.get('production', ['No specific recommendations'])])}
                </ul>
            </div>
            
            <div class="recommendation-category">
                <h3>🎵 Arrangement Improvements</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in recommendations.get('arrangement', ['No specific recommendations'])])}
                </ul>
            </div>
            
            <div class="recommendation-category">
                <h3>✍️ Lyrical Enhancements</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in recommendations.get('lyrics', ['No specific recommendations'])])}
                </ul>
            </div>
            
            <div class="recommendation-category">
                <h3>🚀 Immediate Actions</h3>
                <ul>
                    {''.join([f'<li>{item}</li>' for item in recommendations.get('immediate_actions', ['No immediate actions required'])])}
                </ul>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>Detailed Analysis Summary</h2>
        
        <h3>Musical Meaning & Character</h3>
        <div class="key-insight">
            <strong>Emotional Landscape:</strong>
            {report_data.get('detailed_analysis', {}).get('musical_meaning', {}).get('emotional_landscape', {}).get('primary_emotion', 'Analysis pending')}
        </div>
        
        <h3>Commercial Viability</h3>
        <div class="key-insight">
            <strong>Market Positioning:</strong>
            {report_data.get('detailed_analysis', {}).get('hit_comparison', {}).get('market_positioning', {}).get('positioning_strategy', 'Analysis pending')}
        </div>
        
        <h3>Innovation Assessment</h3>
        <div class="key-insight">
            <strong>Differentiation Factors:</strong>
            {', '.join(report_data.get('detailed_analysis', {}).get('novelty_assessment', {}).get('differentiation_analysis', {}).get('unique_selling_points', ['Analysis pending']))}
        </div>
    </div>
    
    <footer style="text-align: center; margin-top: 50px; padding: 20px; border-top: 1px solid #ddd; color: #7f8c8d;">
        <p>Generated by ChartMuse AI Analysis Engine v{metadata['report_version']} | {metadata['analysis_date']}</p>
        <p><strong>Confidential:</strong> This report contains proprietary analysis and should not be distributed without authorization.</p>
    </footer>
</body>
</html>
            """
            
            # Save HTML file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Sanitize filename by removing/replacing special characters
            safe_title = ''.join(c if c.isalnum() or c in '-_' else '_' for c in metadata['song_title'])
            filename = f"analysis_report_{safe_title}_{timestamp}.html"
            filepath = self.output_dir / filename
            
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating HTML report: {e}")
            raise
    
    def _generate_pdf_report(self, report_data: Dict[str, Any]) -> str:
        """Generate professional PDF report"""
        if not PDF_AVAILABLE:
            logger.warning("PDF generation not available - ReportLab not installed")
            return ""
        
        try:
            metadata = report_data['metadata']
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # Sanitize filename by removing/replacing special characters
            safe_title = ''.join(c if c.isalnum() or c in '-_' else '_' for c in metadata['song_title'])
            filename = f"analysis_report_{safe_title}_{timestamp}.pdf"
            filepath = self.output_dir / filename
            
            # Create PDF document
            doc = SimpleDocTemplate(
                str(filepath),
                pagesize=letter,
                rightMargin=72,
                leftMargin=72,
                topMargin=72,
                bottomMargin=72
            )
            
            story = []
            
            # Title page
            story.append(Paragraph(f"Professional Music Analysis Report", self.styles['Title']))
            story.append(Spacer(1, 0.2*inch))
            story.append(Paragraph(f"{metadata['song_title']}", self.styles['Heading1']))
            story.append(Paragraph(f"by {metadata['artist']}", self.styles['Heading2']))
            story.append(Spacer(1, 0.3*inch))
            story.append(Paragraph(f"Analysis Date: {metadata['analysis_date']}", self.styles['Normal']))
            story.append(Paragraph(f"Report Version: {metadata['report_version']}", self.styles['Normal']))
            story.append(PageBreak())
            
            # Executive Summary
            executive_summary = report_data['executive_summary']
            story.append(Paragraph("Executive Summary", self.styles['SectionHeader']))
            story.append(Paragraph(
                f"Commercial Potential: <b>{self._format_commercial_potential_text(executive_summary.get('commercial_potential', 'Medium'))}</b>",
                self.styles['ExecutiveSummary']
            ))
            story.append(Paragraph(
                f"Confidence Level: <b>{self._safe_string_format(executive_summary.get('confidence_level', 'N/A'))}</b>",
                self.styles['ExecutiveSummary']
            ))
            story.append(Paragraph(
                f"Investment Recommendation: {self._safe_string_format(executive_summary.get('investment_recommendation', 'Analysis pending'))}",
                self.styles['ExecutiveSummary']
            ))
            story.append(Spacer(1, 0.3*inch))
            
            # Key Strengths and Concerns
            if executive_summary.get('key_strengths'):
                story.append(Paragraph("Key Strengths:", self.styles['Heading3']))
                for strength in executive_summary['key_strengths']:
                    story.append(Paragraph(f"• {strength}", self.styles['Recommendation']))
                story.append(Spacer(1, 0.2*inch))
            
            if executive_summary.get('primary_concerns'):
                story.append(Paragraph("Primary Concerns:", self.styles['Heading3']))
                for concern in executive_summary['primary_concerns']:
                    story.append(Paragraph(f"• {concern}", self.styles['Recommendation']))
                story.append(Spacer(1, 0.2*inch))
            
            # Strategic Recommendations
            story.append(Paragraph("Strategic Recommendations", self.styles['SectionHeader']))
            recommendations = report_data['recommendations']
            
            for category, items in recommendations.items():
                if items and category != 'strategic':
                    story.append(Paragraph(f"{category.title()} Recommendations:", self.styles['Heading3']))
                    for item in items[:5]:  # Limit to top 5 per category
                        story.append(Paragraph(f"• {item}", self.styles['Recommendation']))
                    story.append(Spacer(1, 0.1*inch))
            
            # Build PDF
            doc.build(story)
            
            logger.info(f"Generated PDF report: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating PDF report: {e}")
            return ""
    
    def _generate_json_export(self, analysis_results: Dict[str, Any], song_metadata: Dict[str, Any]) -> str:
        """Generate JSON export for programmatic access"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_data_{song_metadata.get('title', 'unknown').replace(' ', '_')}_{timestamp}.json"
            filepath = self.output_dir / filename
            
            export_data = {
                'metadata': {
                    'song_title': song_metadata.get('title', 'Unknown'),
                    'artist': song_metadata.get('artist', 'Unknown'),
                    'genre': song_metadata.get('genre', 'Unknown'),
                    'analysis_timestamp': datetime.now().isoformat(),
                    'export_version': '2.0.0'
                },
                'analysis_results': analysis_results,
                'song_metadata': song_metadata
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Generated JSON export: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error generating JSON export: {e}")
            return ""
    
    def _create_simple_chart(self, title: str, message: str) -> str:
        """Create a simple text-based chart as fallback"""
        try:
            fig = go.Figure()
            fig.add_annotation(
                text=f"<b>{title}</b><br><br>{message}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False,
                font=dict(size=16, color="darkblue")
            )
            fig.update_layout(
                width=600, height=400,
                title=title,
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                plot_bgcolor='white'
            )
            return fig.to_html(include_plotlyjs='inline', div_id=f"chart_{title.lower().replace(' ', '_')}")
        except Exception as e:
            logger.warning(f"Could not create simple chart: {e}")
            return f"<div><h3>{title}</h3><p>{message}</p></div>"
    
    def _create_fallback_radar_chart(self, analysis_results: Dict[str, Any]) -> str:
        """Create a fallback radar chart from available data"""
        try:
            # Use overall assessment scores if available
            overall_assessment = analysis_results.get('overall_assessment', {})
            
            categories = ['Commercial', 'Artistic', 'Innovation', 'Technical']
            values = [
                overall_assessment.get('commercial_score', 0.5),
                overall_assessment.get('artistic_score', 0.5), 
                overall_assessment.get('innovation_score', 0.5),
                0.75  # Default technical score
            ]
            
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Analysis Scores'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 1]
                    )),
                showlegend=True,
                title="Analysis Overview",
                width=500, height=400
            )
            
            return fig.to_html(include_plotlyjs='inline', div_id="fallback_radar_chart")
        except Exception as e:
            logger.warning(f"Fallback radar chart failed: {e}")
            return self._create_simple_chart("Feature Analysis", "Analysis data unavailable")
    
    def _create_simple_probability_chart(self, score: float) -> str:
        """Create a simple probability gauge chart"""
        try:
            fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Hit Probability %"},
                gauge = {'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 25], 'color': "lightgray"},
                            {'range': [25, 50], 'color': "gray"},
                            {'range': [50, 75], 'color': "lightgreen"},
                            {'range': [75, 100], 'color': "green"}],
                        'threshold': {'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75, 'value': 90}}))
            
            fig.update_layout(width=500, height=400)
            return fig.to_html(include_plotlyjs='inline', div_id="simple_probability_chart")
        except Exception as e:
            logger.warning(f"Simple probability chart failed: {e}")
            return self._create_simple_chart("Hit Probability", f"Score: {score*100:.1f}%")
    
    def _format_commercial_potential_class(self, value) -> str:
        """Convert commercial potential value to CSS class name"""
        try:
            if isinstance(value, (int, float)):
                # Convert numeric value to text category
                if value >= 0.8:
                    return "high"
                elif value >= 0.6:
                    return "medium"
                else:
                    return "low"
            elif isinstance(value, str):
                return value.lower()
            else:
                return "medium"
        except:
            return "medium"
    
    def _format_commercial_potential_text(self, value) -> str:
        """Convert commercial potential value to display text"""
        try:
            if isinstance(value, (int, float)):
                # Convert numeric value to text category
                if value >= 0.8:
                    return "High"
                elif value >= 0.6:
                    return "Medium"
                else:
                    return "Low"
            elif isinstance(value, str):
                return value.title()
            else:
                return "Medium"
        except:
            return "Medium"
    
    def _safe_string_format(self, value) -> str:
        """Safely convert any value to string for template formatting"""
        try:
            if isinstance(value, (int, float)):
                # Format numeric values appropriately
                if isinstance(value, float) and 0 <= value <= 1:
                    # Likely a percentage/score, format as percentage
                    return f"{value * 100:.1f}%"
                else:
                    return str(value)
            elif isinstance(value, str):
                return value
            elif value is None:
                return "N/A"
            else:
                return str(value)
        except:
            return "N/A"