"""
Model factory for creating models based on configuration
"""

from .mlp import MLPModel
from .residual_mlp import ResidualMLPModel

try:  # Optional dependency for environments without the MIRT model file.
    from .mirt import MIRTModel  # type: ignore
except ImportError:  # pragma: no cover
    MIRTModel = None  # type: ignore

def create_model(config, embedding_dim, num_models=None):
    """
    Create model based on configuration
    
    Args:
        config: Configuration dictionary
        embedding_dim: Input embedding dimension
        
    Returns:
        Model instance
    """
    pipeline_config = config['pipeline_config']
    model_type = pipeline_config['model_type']
    model_config = pipeline_config['model_config']
    
    if model_type == "mlp":
        model = MLPModel(
            input_dim=embedding_dim,
            hidden_dims=model_config['hidden_dims'],
            output_dim=model_config['output_dim'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            use_batch_norm=model_config.get('use_batch_norm', False),
            use_layer_norm=model_config.get('use_layer_norm', False),
            use_weight_decay=model_config.get('use_weight_decay', True),
            weight_decay=model_config.get('weight_decay', 1e-4),
            use_spectral_norm=model_config.get('use_spectral_norm', False),
            use_dropout2d=model_config.get('use_dropout2d', False)
        )
    elif model_type == "residual_mlp":
        model = ResidualMLPModel(
            input_dim=embedding_dim,
            hidden_dims=model_config['hidden_dims'],
            output_dim=model_config['output_dim'],
            dropout=model_config['dropout'],
            activation=model_config['activation'],
            num_blocks=model_config.get('num_blocks', 3),
            use_batch_norm=model_config.get('use_batch_norm', False),
            use_layer_norm=model_config.get('use_layer_norm', True),
            use_weight_decay=model_config.get('use_weight_decay', True),
            weight_decay=model_config.get('weight_decay', 1e-4),
            use_spectral_norm=model_config.get('use_spectral_norm', False)
        )
    elif model_type == "mirt":
        if MIRTModel is None:
            raise ImportError("MIRTModel is not available. Ensure models/mirt.py is present.")
        if num_models is None:
            raise ValueError("num_models must be provided when using the MIRT model.")
        model = MIRTModel(
            num_llms=num_models,
            llm_embedding_dim=model_config.get('llm_embedding_dim', 128),
            item_input_dim=embedding_dim,
            latent_dim=model_config.get('latent_dim', 16),
            a_range=model_config.get('a_range'),
            theta_range=model_config.get('theta_range'),
            irf_kwargs=model_config.get('irf_kwargs', {})
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}. Supported types: 'mlp', 'residual_mlp'")
    
    return model
