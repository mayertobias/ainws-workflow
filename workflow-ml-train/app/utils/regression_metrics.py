"""
Business-Relevant Regression Evaluation Metrics for Hit Song Science

Provides specialized metrics for evaluating regression models in the context
of hit song prediction, beyond standard regression metrics.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

logger = logging.getLogger(__name__)

class HitSongRegressionMetrics:
    """
    Business-relevant metrics for regression-based hit song prediction.
    
    Provides metrics that matter to music industry professionals:
    - Ranking accuracy (how well does model rank songs?)
    - Top-K precision (accuracy in identifying actual hits)
    - Threshold analysis (performance at different hit cutoffs)
    - Percentile accuracy (accuracy within score ranges)
    """
    
    def __init__(self, hit_thresholds: Optional[List[float]] = None):
        """
        Initialize the regression metrics calculator.
        
        Args:
            hit_thresholds: List of hit score thresholds to evaluate (0.0-1.0)
                           Default: [0.6, 0.7, 0.8, 0.9]
        """
        self.hit_thresholds = hit_thresholds or [0.6, 0.7, 0.8, 0.9]
        logger.info(f"Initialized HitSongRegressionMetrics with thresholds: {self.hit_thresholds}")
    
    def calculate_comprehensive_metrics(self, 
                                      y_true: np.ndarray, 
                                      y_pred: np.ndarray,
                                      feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive regression metrics for hit song prediction.
        
        Args:
            y_true: True hit scores (0-1 range)
            y_pred: Predicted hit scores (0-1 range)
            feature_names: Optional feature names for analysis
            
        Returns:
            Dictionary of comprehensive metrics
        """
        logger.info(f"Calculating comprehensive regression metrics for {len(y_true)} predictions")
        
        # Ensure arrays are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Validate inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"Length mismatch: y_true ({len(y_true)}) != y_pred ({len(y_pred)})")
        
        metrics = {}
        
        # 1. Standard Regression Metrics
        metrics['standard_regression'] = self._calculate_standard_metrics(y_true, y_pred)
        
        # 2. Ranking Metrics (Critical for Music Industry)
        metrics['ranking_performance'] = self._calculate_ranking_metrics(y_true, y_pred)
        
        # 3. Top-K Precision (Hit Identification)
        metrics['top_k_precision'] = self._calculate_top_k_precision(y_true, y_pred)
        
        # 4. Threshold Analysis (Hit/Miss at Different Cutoffs)
        metrics['threshold_analysis'] = self._calculate_threshold_analysis(y_true, y_pred)
        
        # 5. Percentile Accuracy (Score Range Performance)
        metrics['percentile_accuracy'] = self._calculate_percentile_accuracy(y_true, y_pred)
        
        # 6. Distribution Analysis
        metrics['distribution_analysis'] = self._calculate_distribution_analysis(y_true, y_pred)
        
        # 7. Business Impact Metrics
        metrics['business_impact'] = self._calculate_business_impact_metrics(y_true, y_pred)
        
        # 8. Model Calibration
        metrics['calibration'] = self._calculate_calibration_metrics(y_true, y_pred)
        
        # 9. Summary Statistics
        metrics['summary'] = self._create_summary_statistics(y_true, y_pred, metrics)
        
        logger.info(f"✅ Comprehensive metrics calculated:")
        logger.info(f"   📊 R² Score: {metrics['standard_regression']['r2_score']:.3f}")
        logger.info(f"   📊 Ranking Correlation: {metrics['ranking_performance']['spearman_correlation']:.3f}")
        logger.info(f"   📊 Top-10% Precision: {metrics['top_k_precision']['top_10_percent']:.3f}")
        logger.info(f"   📊 Hit Detection (0.8): {metrics['threshold_analysis'][0.8]['precision']:.3f}")
        
        return metrics
    
    def _calculate_standard_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate standard regression metrics."""
        
        # Handle edge cases
        if len(np.unique(y_true)) == 1:
            logger.warning("All true values are identical, R² will be undefined")
            r2 = 0.0
        else:
            r2 = r2_score(y_true, y_pred)
        
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        
        # Mean Absolute Percentage Error (handle division by zero)
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100
        
        # Explained Variance Score
        explained_variance = 1 - (np.var(y_true - y_pred) / np.var(y_true))
        
        return {
            'r2_score': float(r2),
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'explained_variance': float(explained_variance)
        }
    
    def _calculate_ranking_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate ranking-based metrics (critical for music industry)."""
        
        # Spearman correlation (rank-based)
        spearman_corr, spearman_p = stats.spearmanr(y_true, y_pred)
        
        # Kendall's tau (rank-based)
        kendall_tau, kendall_p = stats.kendalltau(y_true, y_pred)
        
        # Pearson correlation (linear relationship)
        pearson_corr, pearson_p = stats.pearsonr(y_true, y_pred)
        
        # Ranking accuracy (how often model ranks song A > song B correctly)
        ranking_accuracy = self._calculate_pairwise_ranking_accuracy(y_true, y_pred)
        
        return {
            'spearman_correlation': float(spearman_corr),
            'spearman_p_value': float(spearman_p),
            'kendall_tau': float(kendall_tau),
            'kendall_p_value': float(kendall_p),
            'pearson_correlation': float(pearson_corr),
            'pearson_p_value': float(pearson_p),
            'pairwise_ranking_accuracy': float(ranking_accuracy)
        }
    
    def _calculate_top_k_precision(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate top-K precision metrics."""
        
        n_samples = len(y_true)
        top_k_metrics = {}
        
        # Different percentiles for top-K analysis
        percentiles = [1, 5, 10, 20, 25]
        
        for percentile in percentiles:
            k = max(1, int(n_samples * percentile / 100))
            
            # Get top-K predicted indices
            top_k_pred_indices = np.argsort(y_pred)[-k:]
            
            # Get top-K actual indices
            top_k_true_indices = np.argsort(y_true)[-k:]
            
            # Calculate precision: how many of predicted top-K are actually in true top-K
            intersection = len(set(top_k_pred_indices) & set(top_k_true_indices))
            precision = intersection / k if k > 0 else 0.0
            
            top_k_metrics[f'top_{percentile}_percent'] = float(precision)
        
        return top_k_metrics
    
    def _calculate_threshold_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[float, Dict[str, float]]:
        """Calculate performance at different hit score thresholds."""
        
        threshold_metrics = {}
        
        for threshold in self.hit_thresholds:
            # Convert to binary at threshold
            y_true_binary = (y_true >= threshold).astype(int)
            y_pred_binary = (y_pred >= threshold).astype(int)
            
            # Calculate classification metrics at threshold
            tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
            fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
            tn = np.sum((y_true_binary == 0) & (y_pred_binary == 0))
            fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
            
            # Calculate metrics
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
            
            threshold_metrics[threshold] = {
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'accuracy': float(accuracy),
                'true_positives': int(tp),
                'false_positives': int(fp),
                'true_negatives': int(tn),
                'false_negatives': int(fn)
            }
        
        return threshold_metrics
    
    def _calculate_percentile_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate accuracy within different score percentiles."""
        
        percentile_ranges = [
            (0, 20, 'bottom_20'),
            (20, 40, 'low_40'),
            (40, 60, 'middle_60'),
            (60, 80, 'high_80'),
            (80, 100, 'top_20')
        ]
        
        percentile_metrics = {}
        
        for low, high, label in percentile_ranges:
            # Get percentile boundaries from true values
            low_bound = np.percentile(y_true, low)
            high_bound = np.percentile(y_true, high)
            
            # Find samples in this percentile range
            mask = (y_true >= low_bound) & (y_true <= high_bound)
            
            if np.sum(mask) > 0:
                # Calculate MAE for this percentile
                mae_percentile = np.mean(np.abs(y_true[mask] - y_pred[mask]))
                
                # Calculate correlation for this percentile
                if np.sum(mask) > 2:  # Need at least 3 points for correlation
                    corr_percentile = np.corrcoef(y_true[mask], y_pred[mask])[0, 1]
                    corr_percentile = corr_percentile if not np.isnan(corr_percentile) else 0.0
                else:
                    corr_percentile = 0.0
                
                percentile_metrics[label] = {
                    'mae': float(mae_percentile),
                    'correlation': float(corr_percentile),
                    'sample_count': int(np.sum(mask)),
                    'score_range': f'{low_bound:.3f}-{high_bound:.3f}'
                }
        
        return percentile_metrics
    
    def _calculate_distribution_analysis(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction and residual distributions."""
        
        residuals = y_true - y_pred
        
        return {
            'residuals': {
                'mean': float(np.mean(residuals)),
                'std': float(np.std(residuals)),
                'skewness': float(stats.skew(residuals)),
                'kurtosis': float(stats.kurtosis(residuals)),
                'min': float(np.min(residuals)),
                'max': float(np.max(residuals))
            },
            'predictions': {
                'mean': float(np.mean(y_pred)),
                'std': float(np.std(y_pred)),
                'min': float(np.min(y_pred)),
                'max': float(np.max(y_pred)),
                'coverage_0_1': float(np.mean((y_pred >= 0) & (y_pred <= 1)))
            },
            'true_values': {
                'mean': float(np.mean(y_true)),
                'std': float(np.std(y_true)),
                'min': float(np.min(y_true)),
                'max': float(np.max(y_true))
            }
        }
    
    def _calculate_business_impact_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate business impact metrics for music industry."""
        
        # Simulate business scenarios
        # Top 10% selection accuracy (A&R use case)
        top_10_percent_count = max(1, int(len(y_true) * 0.1))
        top_10_pred_indices = np.argsort(y_pred)[-top_10_percent_count:]
        top_10_true_indices = np.argsort(y_true)[-top_10_percent_count:]
        
        # A&R Success Rate: If we sign top 10% predicted, what's our hit rate?
        ar_success_rate = np.mean(y_true[top_10_pred_indices])
        
        # Missed Opportunity: How many actual hits did we miss?
        missed_hits = len(set(top_10_true_indices) - set(top_10_pred_indices))
        
        # False Positives: How many predicted hits weren't actually hits?
        false_positives = len(set(top_10_pred_indices) - set(top_10_true_indices))
        
        # Value at Risk: Average score of songs we incorrectly predicted as hits
        if false_positives > 0:
            false_positive_indices = list(set(top_10_pred_indices) - set(top_10_true_indices))
            value_at_risk = np.mean(y_true[false_positive_indices])
        else:
            value_at_risk = 0.0
        
        return {
            'ar_success_rate': float(ar_success_rate),
            'missed_opportunities': int(missed_hits),
            'false_positives': int(false_positives),
            'value_at_risk': float(value_at_risk),
            'top_10_overlap': float(len(set(top_10_pred_indices) & set(top_10_true_indices)) / top_10_percent_count)
        }
    
    def _calculate_calibration_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate model calibration metrics."""
        
        # Bin predictions and check if predicted probability matches actual rate
        n_bins = 10
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        
        calibration_error = 0.0
        reliability_scores = []
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Find predictions in this bin
            mask = (y_pred >= bin_lower) & (y_pred < bin_upper)
            
            if np.sum(mask) > 0:
                # Average predicted score in bin
                bin_pred_mean = np.mean(y_pred[mask])
                
                # Average actual score in bin
                bin_true_mean = np.mean(y_true[mask])
                
                # Calibration error for this bin
                bin_error = abs(bin_pred_mean - bin_true_mean)
                calibration_error += bin_error * np.sum(mask)
                
                reliability_scores.append(bin_error)
        
        # Expected Calibration Error
        calibration_error = calibration_error / len(y_pred)
        
        return {
            'expected_calibration_error': float(calibration_error),
            'reliability_score': float(np.mean(reliability_scores)) if reliability_scores else 0.0
        }
    
    def _calculate_pairwise_ranking_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate pairwise ranking accuracy."""
        
        n_samples = len(y_true)
        if n_samples < 2:
            return 1.0
        
        # Sample pairs for efficiency (if dataset is large)
        max_pairs = min(10000, n_samples * (n_samples - 1) // 2)
        
        correct_rankings = 0
        total_pairs = 0
        
        # Generate random pairs
        np.random.seed(42)  # For reproducibility
        
        for _ in range(max_pairs):
            i, j = np.random.choice(n_samples, 2, replace=False)
            
            # Check if ranking is correct
            true_ranking = y_true[i] > y_true[j]
            pred_ranking = y_pred[i] > y_pred[j]
            
            if true_ranking == pred_ranking:
                correct_rankings += 1
            
            total_pairs += 1
        
        return correct_rankings / total_pairs if total_pairs > 0 else 0.0
    
    def _create_summary_statistics(self, y_true: np.ndarray, y_pred: np.ndarray, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary statistics for easy interpretation."""
        
        return {
            'overall_performance': {
                'r2_score': metrics['standard_regression']['r2_score'],
                'ranking_correlation': metrics['ranking_performance']['spearman_correlation'],
                'top_10_precision': metrics['top_k_precision']['top_10_percent'],
                'business_success_rate': metrics['business_impact']['ar_success_rate']
            },
            'model_quality': {
                'calibration_error': metrics['calibration']['expected_calibration_error'],
                'prediction_stability': 1 - metrics['distribution_analysis']['residuals']['std'],
                'coverage_validity': metrics['distribution_analysis']['predictions']['coverage_0_1']
            },
            'business_readiness': {
                'hit_identification_80': metrics['threshold_analysis'][0.8]['precision'],
                'missed_opportunities': metrics['business_impact']['missed_opportunities'],
                'false_positive_rate': metrics['business_impact']['false_positives']
            }
        }

# Convenience function for easy usage
def evaluate_hit_song_regression(y_true: np.ndarray, 
                                y_pred: np.ndarray,
                                hit_thresholds: Optional[List[float]] = None) -> Dict[str, Any]:
    """
    Convenience function to evaluate hit song regression model.
    
    Args:
        y_true: True hit scores (0-1 range)
        y_pred: Predicted hit scores (0-1 range)
        hit_thresholds: Optional hit score thresholds
        
    Returns:
        Comprehensive evaluation metrics
    """
    evaluator = HitSongRegressionMetrics(hit_thresholds=hit_thresholds)
    return evaluator.calculate_comprehensive_metrics(y_true, y_pred)