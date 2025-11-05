"""
Plotting Module
Generates visualizations for training progress, feature importance, and results.
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import io
import base64
from typing import Dict, Any, Optional, List


def plot_feature_importance(
    feature_importance: Dict[str, float],
    top_n: int = 10,
    figsize: tuple = (8, 6)
) -> str:
    """
    Create a horizontal bar chart of feature importance.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        top_n: Number of top features to display
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded image string
    """
    # Sort features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    feature_names = [f[0] for f in top_features]
    importance_scores = [f[1] for f in top_features]
    
    # Format feature names for readability
    formatted_names = [_format_feature_name(name) for name in feature_names]
    
    # Create plot
    fig, ax = plt.subplots(figsize=figsize)
    ind = range(len(formatted_names))
    
    ax.barh(ind, importance_scores, align='center', color='#c44e52', alpha=0.9)
    ax.set_yticks(ind)
    ax.set_yticklabels(formatted_names)
    ax.set_xlabel('Feature Importance Coefficient', fontsize=12)
    ax.set_title(f'Top {top_n} Features for Prediction', fontsize=14, fontweight='bold')
    ax.tick_params(left=False, top=False, right=False)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def plot_prediction_comparison(
    actual: List[float],
    predicted: List[float],
    figsize: tuple = (8, 6)
) -> str:
    """
    Create a scatter plot comparing actual vs predicted values.
    
    Args:
        actual: List of actual target values
        predicted: List of predicted values
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded image string
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(actual, predicted, alpha=0.6, color='#55a868')
    
    # Add diagonal line (perfect prediction)
    min_val = min(min(actual), min(predicted))
    max_val = max(max(actual), max(predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual LOS (days)', fontsize=12)
    ax.set_ylabel('Predicted LOS (days)', fontsize=12)
    ax.set_title('Actual vs Predicted Length of Stay', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def plot_loss_curve(
    loss_history: List[float],
    figsize: tuple = (8, 5)
) -> str:
    """
    Create a line plot of training loss over epochs/iterations.
    
    Args:
        loss_history: List of loss values during training
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded image string
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    epochs = range(1, len(loss_history) + 1)
    ax.plot(epochs, loss_history, 'b-', linewidth=2, label='Training Loss')
    
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss (MSE)', fontsize=12)
    ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def plot_residuals(
    actual: List[float],
    predicted: List[float],
    figsize: tuple = (8, 5)
) -> str:
    """
    Create a residual plot (errors vs predictions).
    
    Args:
        actual: List of actual target values
        predicted: List of predicted values
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded image string
    """
    residuals = np.array(actual) - np.array(predicted)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    ax.scatter(predicted, residuals, alpha=0.6, color='#55a868')
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Predicted LOS (days)', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title('Residual Plot', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return img_base64


def _format_feature_name(feature_name: str) -> str:
    """Format feature names for human readability."""
    name = feature_name.replace('ADM_', '').replace('INS_', '').replace('AGE_', '')
    name = name.replace('_', ' ').title()
    
    replacements = {
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'Neuro SICU',
        'Neuro Intermediate': 'Neuro Intermediate',
        'Other-Icu': 'Other ICU',
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
    
    return name


def plot_uncertainty_distribution(
    predictions: List[float],
    uncertainties: Optional[List[float]] = None,
    figsize: tuple = (8, 5)
) -> str:
    """
    Create a plot showing prediction uncertainty distribution.
    
    Args:
        predictions: List of predicted values
        uncertainties: Optional list of uncertainty estimates
        figsize: Figure size tuple
        
    Returns:
        Base64-encoded image string
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if uncertainties:
        ax.errorbar(range(len(predictions)), predictions, yerr=uncertainties, 
                   fmt='o', alpha=0.6, color='#c44e52', capsize=3)
        ax.set_ylabel('Prediction with Uncertainty Bounds', fontsize=12)
    else:
        ax.hist(predictions, bins=20, color='#55a868', alpha=0.7, edgecolor='black')
        ax.set_ylabel('Frequency', fontsize=12)
    
    ax.set_xlabel('Sample Index' if uncertainties else 'Predicted LOS (days)', fontsize=12)
    ax.set_title('Prediction Uncertainty Distribution', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Convert to base64 string
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png', bbox_inches='tight', dpi=100)
    img_buffer.seek(0)
    img_base64 = base64.b64encode(img_buffer.read()).decode('utf-8')
    plt.close()
    
    return img_base64

