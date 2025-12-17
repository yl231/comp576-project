"""
Standard Multi-Layer Perceptron (MLP) model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .base_model import BaseModel

class MLPBlock(nn.Module):
    """
    MLP block with regularization options
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, activation='relu',
                 use_batch_norm=False, use_layer_norm=False, use_spectral_norm=False):
        super(MLPBlock, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Linear layers with optional spectral normalization
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        
        if use_spectral_norm:
            self.linear1 = nn.utils.spectral_norm(self.linear1)
            self.linear2 = nn.utils.spectral_norm(self.linear2)
        
        # Normalization layers
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        
        if use_batch_norm:
            self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
            self.batch_norm2 = nn.BatchNorm1d(output_dim)
        elif use_layer_norm:
            self.layer_norm1 = nn.LayerNorm(hidden_dim)
            self.layer_norm2 = nn.LayerNorm(output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'swish':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization"""
        for module in [self.linear1, self.linear2]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Forward pass through MLP block with regularization
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # First linear layer
        x = self.linear1(x)
        
        # Apply normalization
        if self.use_batch_norm:
            x = self.batch_norm1(x)
        elif self.use_layer_norm:
            x = self.layer_norm1(x)
        
        # Activation + dropout
        x = self.activation(x)
        x = self.dropout(x)
        
        # Second linear layer
        x = self.linear2(x)
        
        # Apply normalization to output
        if self.use_batch_norm:
            x = self.batch_norm2(x)
        elif self.use_layer_norm:
            x = self.layer_norm2(x)
        
        return x


class MLPModel(BaseModel):
    """
    Multi-layer MLP model for binary classification with regularization
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.1, activation='relu',
                 use_batch_norm=False, use_layer_norm=False, use_weight_decay=True,
                 weight_decay=1e-4, use_spectral_norm=False, use_dropout2d=False):
        """
        Initialize MLP model with regularization options
        
        Args:
            input_dim: Input dimension (embedding dimension)
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension (1 for binary classification)
            dropout: Dropout rate
            activation: Activation function ('relu', 'gelu', 'swish')
            use_batch_norm: Whether to use batch normalization
            use_layer_norm: Whether to use layer normalization
            use_weight_decay: Whether to apply weight decay (L2 regularization)
            weight_decay: Weight decay coefficient
            use_spectral_norm: Whether to use spectral normalization
            use_dropout2d: Whether to use 2D dropout (for structured dropout)
        """
        super(MLPModel, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_decay=use_weight_decay,
            weight_decay=weight_decay,
            use_spectral_norm=use_spectral_norm,
            use_dropout2d=use_dropout2d
        )
        
        # Store regularization parameters
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay
        self.use_spectral_norm = use_spectral_norm
        self.use_dropout2d = use_dropout2d
        
        # Build MLP layers
        self.layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        
        # Hidden layers
        for hidden_dim in hidden_dims:
            self.layers.append(MLPBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_spectral_norm=use_spectral_norm
            ))
            prev_dim = hidden_dim
        
        # Output layer with optional spectral normalization
        self.output_layer = nn.Linear(prev_dim, output_dim)
        if use_spectral_norm:
            self.output_layer = nn.utils.spectral_norm(self.output_layer)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x):
        """
        Forward pass through the model
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, output_dim)
        """
        # Pass through hidden layers
        for layer in self.layers:
            x = layer(x)
        
        # Output layer
        x = self.output_layer(x)
        
        return x
    
    def get_embeddings(self, x, layer_idx=-1):
        """
        Get intermediate embeddings from a specific layer
        
        Args:
            x: Input tensor
            layer_idx: Index of layer to extract embeddings from (-1 for last hidden layer)
            
        Returns:
            Embeddings tensor
        """
        with torch.no_grad():
            for i, layer in enumerate(self.layers):
                if i == layer_idx:
                    # Return embeddings before the output layer
                    x = layer.linear1(x)
                    x = layer.activation(x)
                    return x
                x = layer(x)
        
        return x
    
    def get_regularization_loss(self):
        """
        Compute regularization loss (L2 weight decay)
        
        Returns:
            Regularization loss tensor
        """
        if not self.use_weight_decay:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        l2_loss = 0.0
        for param in self.parameters():
            if param.requires_grad:
                l2_loss += torch.sum(param ** 2)
        
        return self.weight_decay * l2_loss
    
    def get_model_info(self):
        """Get model information with additional MLP-specific info"""
        info = super().get_model_info()
        info['layers'] = len(self.layers) + 1  # +1 for output layer
        info['regularization'] = {
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_weight_decay': self.use_weight_decay,
            'weight_decay': self.weight_decay,
            'use_spectral_norm': self.use_spectral_norm,
            'use_dropout2d': self.use_dropout2d
        }
        return info
