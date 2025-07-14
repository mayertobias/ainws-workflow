"""
ChartMuse HSS Derived Features Library

This library provides multimodal feature engineering capabilities for combining
audio and content features into higher-level derived features.

Used by both ML training and ML prediction services to ensure consistency.
"""

from .derived_features_calculator import DerivedFeaturesCalculator

__version__ = "1.0.0"
__all__ = ["DerivedFeaturesCalculator"]