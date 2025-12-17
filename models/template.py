"""
Template for creating new model architectures

To add a new model:
1. Copy this template to a new file (e.g., 'transformer.py')
2. Rename the class to your model name
3. Implement the forward method (required)
4. Override other methods as needed
5. Update models/__init__.py to include your new model
6. Update models/model_factory.py to handle your model type
7. Add configuration examples to config_examples.md
"""

import torch
import torch.nn as nn
from .base_model import BaseModel


class TemplateModel(BaseModel):
    """
    Template model class - replace with your actual model
    
    This class inherits from BaseModel and provides all the standard
    functionality. You only need to implement the forward method.
    """
    
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.1, activation='relu', **kwargs):
        """
        Initialize your model
        
        Args:
            input_dim: Input dimension (embedding dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Activation function
            **kwargs: Additional model-specific parameters
        """
        # Call parent constructor
        super(TemplateModel, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            **kwargs
        )
        
        # TODO: Implement your model architecture here
        # Example:
        # self.layers = nn.ModuleList()
        # self.output_layer = nn.Linear(hidden_dims[-1], output_dim)
        
        # Initialize weights
        self.initialize_weights()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # TODO: Implement forward pass
        # Example:
        # for layer in self.layers:
        #     x = layer(x)
        # x = self.output_layer(x)
        # return x
        
        raise NotImplementedError("Implement the forward method")
    
    def get_model_info(self):
        """
        Get model information with additional model-specific info
        
        Override this method to add model-specific information
        """
        info = super().get_model_info()
        # Add any model-specific information
        # info['your_parameter'] = self.your_parameter
        return info
