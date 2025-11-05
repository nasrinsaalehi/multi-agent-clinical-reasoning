"""
Model Testing Module - Part of Inference Agent
Tests trained models and generates predictions with uncertainty estimates.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def test_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
    """
    Test a trained model on test data.
    
    Args:
        model: Trained model (ModelTrainer instance or sklearn model)
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with test results
    """
    # Make predictions
    if hasattr(model, 'predict'):
        y_pred = model.predict(X_test)
    else:
        raise ValueError("Model must have a predict method")
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    # Calculate prediction errors
    errors = y_test - y_pred
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Calculate uncertainty estimate (using prediction variance)
    uncertainty = np.std(y_pred) / np.mean(np.abs(y_pred)) if np.mean(np.abs(y_pred)) > 0 else 0
    
    results = {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mean_error': mean_error,
        'std_error': std_error,
        'uncertainty': uncertainty,
        'predictions': y_pred.tolist(),
        'actual': y_test.tolist(),
        'errors': errors.tolist(),
    }
    
    return results


def predict_single(model, X_single: pd.DataFrame, return_uncertainty: bool = False) -> Dict[str, Any]:
    """
    Make a prediction for a single patient.
    
    Args:
        model: Trained model
        X_single: Single row DataFrame with patient features
        return_uncertainty: Whether to estimate uncertainty
        
    Returns:
        Dictionary with prediction and optional uncertainty
    """
    # Make prediction
    if hasattr(model, 'predict'):
        prediction = model.predict(X_single)[0]
    else:
        raise ValueError("Model must have a predict method")
    
    result = {
        'prediction': float(prediction),
    }
    
    # Estimate uncertainty if requested
    # This is a simplified uncertainty estimate
    # In a real system, you might use Bayesian methods or ensemble variance
    if return_uncertainty:
        # Simple heuristic: use prediction magnitude as uncertainty proxy
        # In practice, you'd use proper uncertainty quantification methods
        uncertainty = abs(prediction) * 0.15  # 15% relative uncertainty
        result['uncertainty'] = float(uncertainty)
        result['prediction_lower'] = float(prediction - uncertainty)
        result['prediction_upper'] = float(prediction + uncertainty)
    
    return result

