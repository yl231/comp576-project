"""
Main workflow pipeline for EmbedLLM training and evaluation
"""

import argparse
import json
import os

import torch

from data import get_data_processor
from models import create_model
from trainer import Trainer
from evaluator import Evaluator

def parse_args():
    parser = argparse.ArgumentParser(description="EmbedLLM training pipeline")
    parser.add_argument(
        "--config",
        "-c",
        default="config.json",
        help="Path to configuration file (default: config.json)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    """
    Main workflow pipeline:
    1. Create/prepare data
    2. Instantiate model
    3. Run training
    4. Run evaluation
    """
    
    # Load configuration
    config_path = args.config
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # 1. Create/prepare data
    print("\n1. Preparing data...")
    data_config = config['data_config']
    dataset_name = data_config['dataset_name']
    text_encoder = data_config['text_encoder']
    
    # Initialize data processor
    data_processor = get_data_processor(config)

    # Load training data
    train_labels, train_embeddings, train_texts = data_processor.load_encoded_data("train")
    print(f"Loaded training data: {train_labels.shape}, {train_embeddings.shape}")
    
    # Load validation data if available
    val_loader = None
    try:
        val_labels, val_embeddings, val_texts = data_processor.load_encoded_data("val")
        val_loader = data_processor.prepare_test_data(
            val_labels, val_embeddings, val_texts,
            batch_size=data_config['batch_size']
        )
        print(f"Loaded validation data: {val_labels.shape}, {val_embeddings.shape}")
    except FileNotFoundError:
        print("No validation data found, using only training data")
    
    # Load test data
    test_labels, test_embeddings, test_texts = data_processor.load_encoded_data("test")
    print(f"Loaded test data: {test_labels.shape}, {test_embeddings.shape}")
    
    # Prepare training and test data
    train_data = data_processor.prepare_training_data(
        train_labels, train_embeddings, train_texts,
        batch_size=data_config['batch_size'],
        shuffle=data_config['shuffle']
    )
    
    test_loader = data_processor.prepare_test_data(
        test_labels, test_embeddings, test_texts,
        batch_size=data_config['batch_size']
    )
    
    print(f"Training batches: {len(train_data['train_loader'])}")
    if val_loader:
        print(f"Validation batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # 2. Instantiate model
    print("\n2. Creating model...")
    embedding_dim = train_embeddings.shape[1]
    
    num_models = train_labels.shape[0]
    model = create_model(config, embedding_dim, num_models=num_models)
    
    requested_device = config['training_config'].get('device', 'cpu')
    if requested_device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(requested_device)
    model = model.to(device)
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"Input dimension: {embedding_dim}")
    
    # 3. Run training
    print("\n3. Starting training...")
    training_config = config['training_config']
    trainer = Trainer(config)
    trainer.model = model
    trainer.device = device
    
    # Initialize optimizer and criterion
    import torch.optim as optim
    import torch.nn as nn
    
    trainer.optimizer = optim.Adam(
        model.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    if config['pipeline_config']['model_type'] == "mirt":
        trainer.criterion = nn.BCELoss()
    else:
        trainer.criterion = nn.BCEWithLogitsLoss()
    
    # Prepare data loaders for training
    data_loaders = {
        'train_loader': train_data['train_loader']
    }
    
    # Add validation loader if available
    if val_loader:
        data_loaders['val_loader'] = val_loader
    else:
        print("No validation data available - training without validation")
    
    # Train the model
    trainer.train(
        data_loaders=data_loaders,
        test_loader=test_loader,
        epochs=training_config['epochs'],
        patience=training_config['patience']
    )
    
    # Save the trained model if enabled
    # if training_config['save_best_model']:
    #     trainer.save_model(training_config['save_paths']['model_path'])
    
    # 4. Run evaluation
    print("\n4. Running evaluation...")
    evaluation_config = config['evaluation_config']
    evaluator = Evaluator(config)
    evaluator.model = model
    evaluator.device = device
    
    # Evaluate on test data
    if evaluation_config['additional_metrics']:
        test_results = evaluator.evaluate_model_performance(test_loader, "EmbedLLM Model")
    else:
        test_results = evaluator.evaluate(test_loader)
    
    # Save results
    results = {
        'test_accuracy': test_results['accuracy'],
        'training_history': {
            'train_losses': trainer.train_losses,
            'train_accuracies': trainer.train_accuracies,
            'val_losses': trainer.val_losses,
            'val_accuracies': trainer.val_accuracies
        },
        'config': config,
        'text_encoder': text_encoder,
        'dataset_name': dataset_name,
        'embedding_dim': embedding_dim
    }
    
    results_path = training_config['save_paths']['results_path']
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nPipeline completed!")
    print(f"Final Test Accuracy: {test_results['accuracy']:.4f}")
    print(f"Results saved to {results_path}")
    
    return results

if __name__ == "__main__":
    main()
