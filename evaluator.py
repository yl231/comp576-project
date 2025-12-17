"""
Evaluator class for model evaluation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

class Evaluator:
    """
    Evaluator class for evaluating trained models
    """
    
    def __init__(self, config):
        """
        Initialize evaluator with configuration
        
        Args:
            config (dict): Configuration dictionary
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Model will be set externally
        self.model = None
    
    def _forward_batch(self, batch):
        if len(batch) == 2:
            batch_X, batch_y = batch
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)
            outputs = self.model(batch_X)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs
        elif len(batch) == 3:
            llm_input, batch_X, batch_y = batch
            batch_X = batch_X.to(self.device)
            batch_y = batch_y.to(self.device)

            if llm_input.dtype in (torch.int64, torch.int32, torch.long):
                llm_input = llm_input.to(self.device)
            else:
                llm_input = llm_input.to(self.device).float()

            outputs = self.model(llm_input, batch_X)
            preds = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            raise ValueError(f"Unexpected batch format with length {len(batch)}")

        return preds, outputs, batch_y

    def evaluate(self, test_loader):
        """
        Evaluate the model on test data
        
        Args:
            test_loader: Test data loader
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not set. Please set self.model before evaluation.")
        
        print("Evaluating model on test data...")
        
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in test_loader:
                preds, outputs, batch_y = self._forward_batch(batch)
                probabilities = preds
                predictions = (probabilities > 0.5).float()
                
                all_predictions.extend(predictions.cpu().numpy().tolist())
                all_labels.extend(batch_y.cpu().numpy().tolist())
                all_probabilities.extend(probabilities.cpu().numpy().tolist())
        
        # Convert to numpy arrays
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        # Calculate overall accuracy (element-wise)
        accuracy = (all_predictions == all_labels).mean()
        
        print(f"Test Accuracy: {accuracy:.4f}")
        
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities
        }
    
    def evaluate_model_performance(self, test_loader, model_name="Model"):
        """
        Comprehensive model performance evaluation
        
        Args:
            test_loader: Test data loader
            model_name: Name of the model for reporting
            
        Returns:
            dict: Comprehensive evaluation results
        """
        results = self.evaluate(test_loader)
        
        # Additional analysis
        predictions = np.array(results['predictions'])
        labels = np.array(results['labels'])
        probabilities = np.array(results['probabilities'])
        
        # Calculate additional metrics
        true_positives = np.sum((predictions == 1) & (labels == 1))
        false_positives = np.sum((predictions == 1) & (labels == 0))
        false_negatives = np.sum((predictions == 0) & (labels == 1))
        true_negatives = np.sum((predictions == 0) & (labels == 0))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Print comprehensive results
        print(f"\n{model_name} Performance Summary:")
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1_score:.4f}")
        print(f"  True Positives: {true_positives}")
        print(f"  False Positives: {false_positives}")
        print(f"  True Negatives: {true_negatives}")
        print(f"  False Negatives: {false_negatives}")
        
        # Add additional metrics to results
        results.update({
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(true_positives),
            'false_positives': int(false_positives),
            'true_negatives': int(true_negatives),
            'false_negatives': int(false_negatives)
        })
        
        return results
