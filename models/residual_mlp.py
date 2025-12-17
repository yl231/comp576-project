"""
Residual Multi-Layer Perceptron (MLP) model with skip connections
"""

import torch
import torch.nn as nn
from .mlp import MLPBlock
from .base_model import BaseModel

class ResidualMLPBlock(nn.Module):
    """
    Residual MLP block with skip connections and regularization
    """
    def __init__(self, dim, hidden_dim, dropout=0.1, activation='relu',
                 use_batch_norm=False, use_layer_norm=True, use_spectral_norm=False):
        super(ResidualMLPBlock, self).__init__()
        
        self.mlp = MLPBlock(dim, hidden_dim, dim, dropout, activation,
                           use_batch_norm=use_batch_norm, use_layer_norm=use_layer_norm,
                           use_spectral_norm=use_spectral_norm)
        
        # Use layer norm by default for residual connections
        if use_layer_norm:
            self.layer_norm = nn.LayerNorm(dim)
        elif use_batch_norm:
            self.batch_norm = nn.BatchNorm1d(dim)
        else:
            self.layer_norm = None
            self.batch_norm = None
        
        self.use_layer_norm = use_layer_norm
        self.use_batch_norm = use_batch_norm
    
    def forward(self, x):
        residual = x
        x = self.mlp(x)
        
        # Apply normalization to residual connection
        if self.use_layer_norm:
            x = self.layer_norm(x + residual)
        elif self.use_batch_norm:
            x = self.batch_norm(x + residual)
        else:
            x = x + residual
        
        return x


class ResidualMLPModel(BaseModel):
    """
    Residual MLP model with skip connections and regularization
    """
    def __init__(self, input_dim, hidden_dims, output_dim=1, dropout=0.1, activation='relu', num_blocks=3,
                 use_batch_norm=False, use_layer_norm=True, use_weight_decay=True,
                 weight_decay=1e-4, use_spectral_norm=False):
        super(ResidualMLPModel, self).__init__(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=dropout,
            activation=activation,
            num_blocks=num_blocks,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_weight_decay=use_weight_decay,
            weight_decay=weight_decay,
            use_spectral_norm=use_spectral_norm
        )
        
        self.num_blocks = num_blocks
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm
        self.use_weight_decay = use_weight_decay
        self.weight_decay = weight_decay
        self.use_spectral_norm = use_spectral_norm
        
        # Input projection with optional spectral normalization
        self.input_proj = nn.Linear(input_dim, hidden_dims[0])
        if use_spectral_norm:
            self.input_proj = nn.utils.spectral_norm(self.input_proj)
        
        # Residual blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            self.blocks.append(ResidualMLPBlock(
                dim=hidden_dims[0],
                hidden_dim=hidden_dims[0],
                dropout=dropout,
                activation=activation,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_spectral_norm=use_spectral_norm
            ))
        
        # Output layer with optional spectral normalization
        self.output_layer = nn.Linear(hidden_dims[0], output_dim)
        if use_spectral_norm:
            self.output_layer = nn.utils.spectral_norm(self.output_layer)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        for module in [self.input_proj, self.output_layer]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """Forward pass"""
        x = self.input_proj(x)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.output_layer(x)
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
        """Get model information with additional ResidualMLP-specific info"""
        info = super().get_model_info()
        info['num_blocks'] = self.num_blocks
        info['layers'] = len(self.blocks) + 2  # +2 for input_proj and output_layer
        info['regularization'] = {
            'use_batch_norm': self.use_batch_norm,
            'use_layer_norm': self.use_layer_norm,
            'use_weight_decay': self.use_weight_decay,
            'weight_decay': self.weight_decay,
            'use_spectral_norm': self.use_spectral_norm
        }
        return info
