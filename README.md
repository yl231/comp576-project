# Router Training Framework

A complete training framework for LLM routing experiments supporting multiple datasets (RouterBench) and model architectures (MLP, Residual MLP, MIRT).

## Quick Start

### 1. Setup Environment

Install dependencies using `uv`:

```bash
cd comp576-routers
uv sync
```

This will install all required dependencies including PyTorch, sentence-transformers, and other packages defined in `pyproject.toml`.

### 2. Configure Training

Edit `config.json` to set your dataset and training parameters:

```json
{
  "data_config": {
    "dataset_name": "EmbedLLM",  // or "RouterBench"
    "text_encoder": "all-mpnet-base-v2",
    "batch_size": 32,
    "shuffle": true
  },
  "pipeline_config": {
    "model_type": "mlp",  // or "mirt"
    "model_config": {
      "hidden_dims": [512, 256],
      "output_dim": 112,
      "dropout": 0.1
    }
  },
  "training_config": {
    "device": "cuda",
    "epochs": 100,
    "patience": 20,
    "learning_rate": 0.001
  }
}
```

### 3. Run Pipeline

Run the complete training and evaluation pipeline:

```bash
uv run python main.py
```

Or specify a custom config file:

```bash
uv run python main.py --config my_config.json
```

The pipeline will automatically:
- Download and process data (if needed)
- Create embeddings using the specified text encoder
- Train the model with early stopping
- Evaluate on test set
- Save results to `results.json`
