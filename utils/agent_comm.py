"""
Agent Communication Module
Handles message passing between agents in the multi-agent system.

This module implements the communication protocol between:
1. Perception Agent (data extraction and normalization)
2. Inference Agent (model training and prediction)
3. Explainability Agent (explanation generation)
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import json


class AgentType(Enum):
    """Types of agents in the system."""
    PERCEPTION = "perception"
    INFERENCE = "inference"
    EXPLAINABILITY = "explainability"


class MessageType(Enum):
    """Types of messages between agents."""
    DATA_EXTRACTED = "data_extracted"
    DATA_PREPROCESSED = "data_preprocessed"
    MODEL_TRAINED = "model_trained"
    PREDICTION_READY = "prediction_ready"
    EXPLANATION_REQUESTED = "explanation_requested"
    EXPLANATION_GENERATED = "explanation_generated"


@dataclass
class AgentMessage:
    """
    Message structure for inter-agent communication.
    
    This follows a simple message-passing pattern that can be extended
    to use proper message queues (e.g., RabbitMQ, Redis) or APIs in production.
    """
    sender: AgentType
    receiver: AgentType
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            'sender': self.sender.value,
            'receiver': self.receiver.value,
            'message_type': self.message_type.value,
            'payload': self.payload,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentMessage':
        """Create message from dictionary."""
        return cls(
            sender=AgentType(data['sender']),
            receiver=AgentType(data['receiver']),
            message_type=MessageType(data['message_type']),
            payload=data['payload'],
            timestamp=data.get('timestamp')
        )


class AgentCommunicator:
    """
    Handles communication between agents.
    
    In a production system, this would interface with:
    - Message queues (RabbitMQ, Kafka)
    - REST APIs
    - gRPC services
    - Event-driven architecture
    
    For now, it uses simple function calls and in-memory message passing.
    """
    
    def __init__(self):
        self.message_history = []
        self.message_handlers = {}
    
    def send_message(self, message: AgentMessage) -> bool:
        """
        Send a message from one agent to another.
        
        Args:
            message: AgentMessage instance
            
        Returns:
            True if message was sent successfully
        """
        # In production: send to message queue or API
        # For now: store in history and trigger handler if registered
        self.message_history.append(message)
        
        handler_key = (message.receiver, message.message_type)
        if handler_key in self.message_handlers:
            handler = self.message_handlers[handler_key]
            handler(message)
        
        return True
    
    def register_handler(
        self, 
        receiver: AgentType, 
        message_type: MessageType, 
        handler_func
    ):
        """
        Register a message handler function.
        
        Args:
            receiver: Target agent type
            message_type: Type of message to handle
            handler_func: Function to call when message is received
        """
        key = (receiver, message_type)
        self.message_handlers[key] = handler_func
    
    def get_message_history(self, agent: Optional[AgentType] = None) -> list:
        """
        Get message history, optionally filtered by agent.
        
        Args:
            agent: Optional agent type to filter by
            
        Returns:
            List of messages
        """
        if agent:
            return [msg for msg in self.message_history if msg.receiver == agent or msg.sender == agent]
        return self.message_history


# Example usage patterns for agent communication:

def perception_to_inference_workflow(data_path: str, communicator: AgentCommunicator):
    """
    Example workflow: Perception Agent extracts data and notifies Inference Agent.
    
    This demonstrates how agents communicate:
    1. Perception Agent extracts and preprocesses data
    2. Sends message to Inference Agent with processed data
    3. Inference Agent can then train models
    """
    from .load_data import load_csv_dataset
    from .preprocessing import preprocess_data
    
    # Perception Agent: Extract data
    df = load_csv_dataset(data_path)
    
    # Perception Agent: Preprocess data
    features_df, target, preprocessing_info = preprocess_data(df)
    
    # Perception Agent: Notify Inference Agent
    message = AgentMessage(
        sender=AgentType.PERCEPTION,
        receiver=AgentType.INFERENCE,
        message_type=MessageType.DATA_PREPROCESSED,
        payload={
            'features_shape': features_df.shape,
            'target_shape': target.shape,
            'preprocessing_info': preprocessing_info
        }
    )
    communicator.send_message(message)
    
    return features_df, target, preprocessing_info


def inference_to_explainability_workflow(
    prediction: float,
    feature_importance: Dict[str, float],
    input_features: Dict[str, Any],
    communicator: AgentCommunicator
):
    """
    Example workflow: Inference Agent makes prediction and requests explanation.
    
    This demonstrates how the Inference Agent can request explanations:
    1. Inference Agent makes prediction
    2. Sends message to Explainability Agent requesting explanation
    3. Explainability Agent generates and returns explanation
    """
    # Inference Agent: Notify Explainability Agent
    message = AgentMessage(
        sender=AgentType.INFERENCE,
        receiver=AgentType.EXPLAINABILITY,
        message_type=MessageType.EXPLANATION_REQUESTED,
        payload={
            'prediction': prediction,
            'feature_importance': feature_importance,
            'input_features': input_features
        }
    )
    communicator.send_message(message)
    
    # In a real async system, the Explainability Agent would process this
    # and send back an EXPLANATION_GENERATED message
    return message

