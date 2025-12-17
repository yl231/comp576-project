"""
Abstract base model class for all model architectures
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, List


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all model architectures
    
    All concrete models must inherit from this class and implement
    the required abstract methods.
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int, 
                 dropout: float = 0.1, activation: str = 'relu', **kwargs):
        """
        Initialize base model
        
        Args:
            input_dim: Input dimension (embedding dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            activation: Activation function name
            **kwargs: Additional model-specific parameters
        """
        super(BaseModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.dropout = dropout
        self.activation = activation
        
        # Store additional parameters
        self.model_params = kwargs
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        pass
    
    def count_parameters(self) -> int:
        """
        Count total number of trainable parameters
        
        Returns:
            Total number of parameters
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary containing model information
        """
        return {
            'model_type': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'dropout': self.dropout,
            'activation': self.activation,
            'total_parameters': self.count_parameters(),
            'model_params': self.model_params
        }
    
    def get_embeddings(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Get intermediate embeddings from a specific layer
        
        Args:
            x: Input tensor
            layer_idx: Index of layer to extract embeddings from (-1 for last hidden layer)
            
        Returns:
            Embeddings tensor
        """
        # Default implementation - can be overridden by subclasses
        with torch.no_grad():
            return self.forward(x)
    
    def freeze_parameters(self, freeze: bool = True) -> None:
        """
        Freeze or unfreeze model parameters
        
        Args:
            freeze: If True, freeze parameters; if False, unfreeze
        """
        for param in self.parameters():
            param.requires_grad = not freeze
    
    def get_layer_weights(self, layer_name: str = None) -> Dict[str, torch.Tensor]:
        """
        Get weights from specific layer or all layers
        
        Args:
            layer_name: Name of specific layer to get weights from
            
        Returns:
            Dictionary of layer weights
        """
        weights = {}
        for name, param in self.named_parameters():
            if layer_name is None or layer_name in name:
                weights[name] = param.data.clone()
        return weights
    
    def set_layer_weights(self, weights: Dict[str, torch.Tensor]) -> None:
        """
        Set weights for specific layers
        
        Args:
            weights: Dictionary of layer weights to set
        """
        for name, param in self.named_parameters():
            if name in weights:
                param.data = weights[name].clone()
    
    def get_activation_function(self, activation: str) -> nn.Module:
        """
        Get activation function module
        
        Args:
            activation: Activation function name
            
        Returns:
            Activation function module
        """
        activation_map = {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            'elu': nn.ELU()
        }
        
        if activation not in activation_map:
            raise ValueError(f"Unknown activation: {activation}. "
                           f"Supported: {list(activation_map.keys())}")
        
        return activation_map[activation]
    
    def initialize_weights(self, method: str = 'xavier_uniform') -> None:
        """
        Initialize model weights using specified method
        
        Args:
            method: Weight initialization method
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if method == 'xavier_uniform':
                    nn.init.xavier_uniform_(module.weight)
                elif method == 'xavier_normal':
                    nn.init.xavier_normal_(module.weight)
                elif method == 'kaiming_uniform':
                    nn.init.kaiming_uniform_(module.weight)
                elif method == 'kaiming_normal':
                    nn.init.kaiming_normal_(module.weight)
                else:
                    raise ValueError(f"Unknown initialization method: {method}")
                
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def __str__(self) -> str:
        """String representation of the model"""
        info = self.get_model_info()
        return (f"{info['model_type']}(input_dim={info['input_dim']}, "
                f"hidden_dims={info['hidden_dims']}, output_dim={info['output_dim']}, "
                f"params={info['total_parameters']:,})")
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
