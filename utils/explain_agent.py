"""
Explainability Agent Module
Generates human-readable explanations for model predictions.

This module provides:
1. Feature-based explanations (which features contributed most)
2. Visual explanations (plots and charts)
3. Natural language explanations (placeholder for LLM integration)
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np


def generate_feature_explanation(
    prediction: float,
    feature_importance: Dict[str, float],
    input_features: Dict[str, Any],
    top_n: int = 5
) -> str:
    """
    Generate a text explanation based on top contributing features.
    
    Args:
        prediction: The model's prediction value
        feature_importance: Dictionary mapping feature names to importance scores
        input_features: Dictionary of input feature values for this patient
        top_n: Number of top features to include in explanation
        
    Returns:
        Human-readable explanation string
    """
    # Get top contributing features
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    # Identify which top features are present in the input
    contributing_factors = []
    for feature_name, importance in top_features:
        # Check if this feature is active in the input
        feature_active = False
        feature_value = None
        
        # Handle different feature naming patterns
        if feature_name in input_features:
            feature_value = input_features[feature_name]
            if isinstance(feature_value, (int, float)) and feature_value > 0:
                feature_active = True
            elif isinstance(feature_value, bool) and feature_value:
                feature_active = True
        
        # Also check for feature name patterns (e.g., ADM_EMERGENCY)
        if not feature_active:
            for key, value in input_features.items():
                if isinstance(value, (int, float, bool)) and value:
                    if feature_name.startswith(key) or key in feature_name:
                        feature_active = True
                        feature_value = value
                        break
        
        if feature_active:
            # Format feature name for readability
            readable_name = _format_feature_name(feature_name)
            contributing_factors.append({
                'name': readable_name,
                'importance': importance,
                'value': feature_value
            })
    
    # Generate explanation text
    explanation_parts = [
        f"The predicted length of stay is {prediction:.2f} days."
    ]
    
    if contributing_factors:
        explanation_parts.append("\nKey contributing factors:")
        for i, factor in enumerate(contributing_factors[:3], 1):
            explanation_parts.append(
                f"{i}. {factor['name']} (importance: {factor['importance']:.3f})"
            )
    
    # Add risk level interpretation
    if prediction < 3:
        explanation_parts.append("\nThis suggests a relatively short hospital stay with lower complication risk.")
    elif prediction < 7:
        explanation_parts.append("\nThis indicates a moderate hospital stay with standard monitoring requirements.")
    else:
        explanation_parts.append("\nThis indicates a longer hospital stay, suggesting more intensive care may be needed.")
    
    return "\n".join(explanation_parts)


def _format_feature_name(feature_name: str) -> str:
    """Format feature names for human readability."""
    # Remove prefixes
    name = feature_name.replace('ADM_', '').replace('INS_', '').replace('AGE_', '')
    
    # Replace underscores with spaces and title case
    name = name.replace('_', ' ').title()
    
    # Special handling for common features
    replacements = {
        'Neuro Surgical Intensive Care Unit (Neuro SICU)': 'Neuro SICU Admission',
        'Neuro Intermediate': 'Neuro Intermediate Care',
        'Other-Icu': 'Other ICU',
        'Circulatory': 'Circulatory System Issues',
        'Respiratory': 'Respiratory System Issues',
        'Infectious': 'Infectious Disease',
        'Neoplasms': 'Neoplasms/Tumors',
    }
    
    for old, new in replacements.items():
        if old in name:
            name = name.replace(old, new)
    
    return name


def generate_llm_explanation(
    prediction: float,
    feature_importance: Dict[str, float],
    input_features: Dict[str, Any],
    model_type: str = "GradientBoosting"
) -> str:
    """
    Generate explanation using LLM (placeholder for future integration).
    
    This is a placeholder function that will be extended to call an LLM API
    (e.g., OpenAI GPT, Anthropic Claude, or local LLM) to generate more
    sophisticated natural language explanations.
    
    Args:
        prediction: The model's prediction value
        feature_importance: Dictionary mapping feature names to importance scores
        input_features: Dictionary of input feature values for this patient
        model_type: Type of model used for prediction
        
    Returns:
        LLM-generated explanation (currently returns placeholder text)
    """
    # TODO: Integrate with LLM API for advanced explanations
    # Example integration pattern:
    # 
    # import openai  # or anthropic, etc.
    # 
    # prompt = f"""
    # Based on the following clinical prediction:
    # - Predicted LOS: {prediction:.2f} days
    # - Top contributing factors: {top_features}
    # - Patient characteristics: {input_features}
    # 
    # Provide a clear, clinician-friendly explanation of this prediction,
    # including reasoning and recommended actions.
    # """
    # 
    # response = openai.ChatCompletion.create(
    #     model="gpt-4",
    #     messages=[{"role": "user", "content": prompt}]
    # )
    # return response.choices[0].message.content
    
    # Placeholder explanation
    explanation = f"""
Based on the clinical prediction model ({model_type}), the patient's predicted length of stay 
is {prediction:.2f} days.

[LLM Explanation Placeholder]
This section will be populated with advanced natural language explanations once LLM integration 
is implemented. The LLM will analyze the contributing factors, provide context-aware reasoning,
and suggest clinical considerations based on the prediction.

To integrate an LLM:
1. Add LLM API credentials to settings
2. Implement prompt engineering for clinical context
3. Add safety filters for medical advice
4. Enable streaming for real-time explanations
"""
    
    return explanation.strip()


def generate_visual_explanation_data(
    feature_importance: Dict[str, float],
    input_features: Dict[str, Any],
    top_n: int = 10
) -> Dict[str, Any]:
    """
    Prepare data for visualization of feature contributions.
    
    Args:
        feature_importance: Dictionary mapping feature names to importance scores
        input_features: Dictionary of input feature values for this patient
        top_n: Number of top features to include
        
    Returns:
        Dictionary with data ready for plotting
    """
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:top_n]
    
    plot_data = {
        'feature_names': [f[0] for f in top_features],
        'importance_scores': [f[1] for f in top_features],
        'feature_active': []
    }
    
    # Check which features are active in the input
    for feature_name, _ in top_features:
        active = False
        if feature_name in input_features:
            val = input_features[feature_name]
            if isinstance(val, (int, float)) and val > 0:
                active = True
            elif isinstance(val, bool) and val:
                active = True
        plot_data['feature_active'].append(active)
    
    return plot_data

