import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import json
import wandb

class Trainer:
    """
    Trainer class for training models
    """
    
    def __init__(self, config):
        """
        Initialize trainer with configuration
        
        Args:
            config (dict): Configuration dictionary containing:
                - model_config: Model architecture parameters
                - device: 'cuda' or 'cpu'
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Initialize model, optimizer, and criterion
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.test_losses = []
        self.test_accuracies = []
        
        # Initialize wandb using config
        wandb_config = config.get('wandb_config', {})
        
        # Generate run name if not specified
        run_name = wandb_config.get('name')
        if run_name is None:
            text_encoder = config['data_config']['text_encoder'].replace('/', '_')
            model_type = config['pipeline_config']['model_type']
            run_name = f"{model_type}_router_{text_encoder}"
        
        # Initialize wandb with error handling
        try:
            wandb.init(
                project=wandb_config.get('project', 'embedllm-router'),
                entity=wandb_config.get('entity'),
                name=run_name,
                tags=wandb_config.get('tags', []),
                notes=wandb_config.get('notes', ''),
                config=config,
                save_code=wandb_config.get('save_code', True),
                resume=wandb_config.get('resume', 'allow')
            )
            self.wandb_enabled = True
            print(f"Wandb initialized: {run_name}")
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Continuing without wandb logging...")
            self.wandb_enabled = False
        
    
    def _forward_batch(self, batch):
        """
        Forward model on a batch that may contain either (X, y) or (llm_ids, X, y).
        """
        if len(batch) == 2:
            batch_X, batch_y = batch
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).float()
            outputs = self.model(batch_X)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs
        elif len(batch) == 3:
            llm_input, batch_X, batch_y = batch
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device).float()

            if llm_input.dtype in (torch.int64, torch.int32, torch.long):
                llm_input = llm_input.to(self.device)
            else:
                llm_input = llm_input.to(self.device).float()

            outputs = self.model(llm_input, batch_X)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            raise ValueError(f"Unexpected batch format with length {len(batch)}")

        return preds, outputs, batch_y

    def train_epoch(self, train_loader):
        """
        Train for one epoch
        
        Args:
            train_loader: Training data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch in train_loader:
            # Forward pass
            self.optimizer.zero_grad()
            preds, outputs, batch_y = self._forward_batch(batch)
            loss = self.criterion(preds, batch_y)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            predictions = (preds > 0.5).float()
            correct += (predictions == batch_y).sum().item()
            total += batch_y.numel()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """
        Validate the model
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in val_loader:
                preds, outputs, batch_y = self._forward_batch(batch)
                loss = self.criterion(preds, batch_y)
                
                total_loss += loss.item()
                predictions = (preds > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.numel()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def test(self, test_loader):
        """
        Test the model
        
        Args:
            test_loader: Test data loader
            
        Returns:
            tuple: (average_loss, accuracy)
        """
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                preds, outputs, batch_y = self._forward_batch(batch)
                loss = self.criterion(preds, batch_y)
                
                total_loss += loss.item()
                predictions = (preds > 0.5).float()
                correct += (predictions == batch_y).sum().item()
                total += batch_y.numel()
        
        avg_loss = total_loss / len(test_loader)
        accuracy = correct / total
        
        return avg_loss, accuracy
    
    def train(self, data_loaders, test_loader=None, epochs=100, patience=10):
        """
        Train the model
        
        Args:
            data_loaders: Dictionary containing train and validation loaders
            test_loader: Test data loader for evaluation every epoch
            epochs: Maximum number of epochs
            patience: Early stopping patience
        """
        print(f"Starting training for {epochs} epochs...")
        
        best_val_acc = 0
        patience_counter = 0
        val_loader = data_loaders.get('val_loader')
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self.train_epoch(data_loaders['train_loader'])
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
            else:
                val_loss, val_acc = None, None
            
            # Testing (if test_loader provided)
            test_loss, test_acc = 0.0, 0.0
            if test_loader is not None:
                test_loss, test_acc = self.test(test_loader)
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            self.test_losses.append(test_loss)
            self.test_accuracies.append(test_acc)
            
            # Log to wandb
            if self.wandb_enabled:
                log_payload = {
                    'epoch': epoch + 1,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc
                }
                if val_loss is not None:
                    log_payload.update({
                        'val_loss': val_loss,
                        'val_acc': val_acc
                    })
                wandb.log(log_payload)
            
            # Consolidated logging
            if test_loader is not None and val_loss is not None:
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            elif test_loader is not None:
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
            elif val_loss is not None:
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            else:
                print(f"Epoch {epoch+1:3d}/{epochs}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            # Early stopping
            if val_acc is not None:
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    patience_counter = 0
                    # Save best model
                    # self.save_model(f"./best_model.pth")
                else:
                    patience_counter += 1
                    
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        if val_loader is not None:
            print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
        else:
            print("Training completed without validation set.")
    

    def save_model(self, path=None):
        """
        Save model to file
        """
        if path is None:
            path = self.config['training_config']['save_paths']['model_path']
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
    
    def load_model(self, path):
        """
        Load model from file
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Model loaded from {path}")
    


