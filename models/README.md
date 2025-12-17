# Models Package

This package contains all model architectures for the EmbedLLM router training framework.

## Structure

```
models/
├── __init__.py          # Package initialization and exports
├── base_model.py        # Abstract base model class
├── model_factory.py     # Model creation factory
├── mlp.py              # Standard MLP model
├── residual_mlp.py     # Residual MLP model
├── template.py         # Template for new models
└── README.md           # This file
```

## Available Models

### 1. MLP Model (`mlp.py`)
- **Type**: `"mlp"`
- **Description**: Standard multi-layer perceptron
- **Parameters**: `hidden_dims`, `output_dim`, `dropout`, `activation`

### 2. Residual MLP Model (`residual_mlp.py`)
- **Type**: `"residual_mlp"`
- **Description**: MLP with residual connections
- **Parameters**: `hidden_dims`, `output_dim`, `dropout`, `activation`, `num_blocks`

## Adding New Models

### Step 1: Create Model File
1. Copy `template.py` to a new file (e.g., `transformer.py`)
2. Rename the class to your model name
3. Implement the required methods

### Step 2: Update Package Exports
Add your model to `models/__init__.py`:
```python
from .your_model import YourModel

__all__ = ['MLPModel', 'ResidualMLPModel', 'YourModel', 'create_model']
```

### Step 3: Update Model Factory
Add your model type to `models/model_factory.py`:
```python
elif model_type == "your_model_type":
    model = YourModel(
        input_dim=embedding_dim,
        # ... other parameters
    )
```

### Step 4: Update Configuration
Add your model configuration to `config_examples.md`:
```json
{
  "model_type": "your_model_type",
  "model_config": {
    "hidden_dims": [512, 256],
    "output_dim": 112,
    "dropout": 0.1,
    "activation": "relu",
    "your_parameter": "value"
  }
}
```

## Base Model Class

All models inherit from `BaseModel` which provides:

### Required Implementation
- `forward(self, x)` - **MUST** be implemented by subclasses

### Inherited Methods (No need to implement)
- `count_parameters(self)` - Return total parameter count
- `get_model_info(self)` - Return comprehensive model information
- `get_embeddings(self, x, layer_idx)` - Extract intermediate embeddings
- `freeze_parameters(self, freeze)` - Freeze/unfreeze model parameters
- `get_layer_weights(self, layer_name)` - Get weights from specific layers
- `set_layer_weights(self, weights)` - Set weights for specific layers
- `initialize_weights(self, method)` - Initialize model weights
- `get_activation_function(self, activation)` - Get activation function module

### Constructor
All models must call the parent constructor:
```python
super(YourModel, self).__init__(
    input_dim=input_dim,
    hidden_dims=hidden_dims,
    output_dim=output_dim,
    dropout=dropout,
    activation=activation,
    **kwargs
)
```

### Example Usage
```python
from models import create_model

# Create model from config
num_models = 112  # number of LLMs in the dataset
model = create_model(config, embedding_dim=768, num_models=num_models)

# Get model info
info = model.get_model_info()
print(f"Parameters: {info['total_parameters']:,}")
```

## Configuration

Models are created using the `create_model()` function which reads from `config.json`:

```json
{
  "pipeline_config": {
    "model_type": "mlp",
    "model_config": {
      "hidden_dims": [512, 256, 128],
      "output_dim": 112,
      "dropout": 0.1,
      "activation": "relu"
    }
  }
}
```

## Best Practices

1. **Consistent Interface**: All models should follow the same interface
2. **Documentation**: Include docstrings for all methods
3. **Error Handling**: Validate input parameters
4. **Weight Initialization**: Use appropriate initialization schemes
5. **Testing**: Test your model with different configurations
