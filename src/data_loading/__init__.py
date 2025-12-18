"""
Data Loading Module

This module handles data acquisition and initial loading.
"""

from .download_kaggle_data import download_dataset, validate_data

__all__ = ['download_dataset', 'validate_data']
