import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, roc_curve, auc,
    classification_report, matthews_corrcoef, cohen_kappa_score
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """
    Comprehensive evaluation with:
    - Confusion matrix
    - Precision/Recall/F1
    - ROC-AUC curves
    - Residual analysis
    - Cross-validation metrics
    """
    
    def __init__(self):
        self.results = {}
    
    def evaluate_classifier(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           y_pred_proba: np.ndarray = None, class_names: List[str] = None) -> Dict:
        """Complete classification evaluation"""
        
        print("\n" + "="*70)
        print("CLASSIFICATION EVALUATION")
        print("="*70)
        
        metrics = {}
        
        try:
            # Basic metrics
            metrics['accuracy'] = accuracy_score(y_true, y_pred)
            metrics['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
            metrics['kappa'] = cohen_kappa_score(y_true, y_pred)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            metrics['confusion_matrix'] = cm
            
            # Handle binary and multiclass cases
            if cm.size == 4:  # Binary classification
                metrics['tn'] = cm[0, 0]
                metrics['fp'] = cm[0, 1]
                metrics['fn'] = cm[1, 0]
                metrics['tp'] = cm[1, 1]
            else:  # Multiclass
                metrics['tn'] = 0
                metrics['fp'] = 0
                metrics['fn'] = 0
                metrics['tp'] = 0
            
            # Sensitivity & Specificity (for binary classification)
            if (metrics['tp'] + metrics['fn']) > 0:
                sensitivity = metrics['tp'] / (metrics['tp'] + metrics['fn'])
            else:
                sensitivity = 0
            
            if (metrics['tn'] + metrics['fp']) > 0:
                specificity = metrics['tn'] / (metrics['tn'] + metrics['fp'])
            else:
                specificity = 0
            
            metrics['sensitivity'] = sensitivity
            metrics['specificity'] = specificity
            
            # ROC-AUC (binary classification only)
            if y_pred_proba is not None and len(np.unique(y_true)) == 2:
                try:
                    metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
                    fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
                    metrics['fpr'] = fpr
                    metrics['tpr'] = tpr
                except Exception as e:
                    logger.warning(f"Could not compute ROC-AUC: {str(e)}")
            
            # Print results
            print(f"\n=== METRICS ===")
            print(f"Accuracy:   {metrics['accuracy']:.4f}")
            print(f"Precision:  {metrics['precision']:.4f}")
            print(f"Recall:     {metrics['recall']:.4f}")
            print(f"F1-Score:   {metrics['f1']:.4f}")
            print(f"MCC:        {metrics['mcc']:.4f}")
            print(f"Kappa:      {metrics['kappa']:.4f}")
            print(f"Sensitivity:{sensitivity:.4f}")
            print(f"Specificity:{specificity:.4f}")
            
            if 'roc_auc' in metrics:
                print(f"ROC-AUC:    {metrics['roc_auc']:.4f}")
            
            print(f"\n=== CONFUSION MATRIX ===")
            print(cm)
            
            print(f"\n=== CLASSIFICATION REPORT ===")
            if class_names is None:
                class_names = [f"Class {i}" for i in range(len(np.unique(y_true)))]
            print(classification_report(y_true, y_pred, target_names=class_names, zero_division=0))
            
        except Exception as e:
            logger.error(f"Error during classification evaluation: {str(e)}")
            print(f"Error: {str(e)}")
        
        print(f"{'='*70}\n")
        
        return metrics
    
    def evaluate_regressor(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """Complete regression evaluation"""
        
        print("\n" + "="*70)
        print("REGRESSION EVALUATION")
        print("="*70)
        
        try:
            residuals = y_true - y_pred
            
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred),
                'residuals': residuals,
                'predictions': y_pred
            }
            
            print(f"\n=== METRICS ===")
            print(f"R2-Score: {metrics['r2']:.4f}")
            print(f"RMSE:     {metrics['rmse']:.4f}")
            print(f"MAE:      {metrics['mae']:.4f}")
            print(f"MSE:      {metrics['mse']:.4f}")
            
            print(f"\n=== RESIDUAL ANALYSIS ===")
            print(f"Mean:     {residuals.mean():.6f} (should be ≈ 0)")
            print(f"Std Dev:  {residuals.std():.6f}")
            print(f"Min:      {residuals.min():.6f}")
            print(f"Max:      {residuals.max():.6f}")
            
        except Exception as e:
            logger.error(f"Error during regression evaluation: {str(e)}")
            print(f"Error: {str(e)}")
            metrics = {}
        
        print(f"{'='*70}\n")
        
        return metrics
    
    def plot_confusion_matrix(self, cm: np.ndarray, class_names: List[str], 
                            filename: str = None):
        """Plot confusion matrix"""
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            im = ax.imshow(cm, cmap='Blues')
            
            ax.set_xticks(np.arange(len(class_names)))
            ax.set_yticks(np.arange(len(class_names)))
            ax.set_xticklabels(class_names, rotation=45)
            ax.set_yticklabels(class_names)
            
            # Add text annotations
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax.text(j, i, str(cm[i, j]), ha='center', va='center', 
                           color='white' if cm[i, j] > cm.max() / 2 else 'black')
            
            ax.set_ylabel('True label')
            ax.set_xlabel('Predicted label')
            ax.set_title('Confusion Matrix')
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                print(f"[OK] Confusion matrix saved to {filename}")
            else:
                plt.show()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting confusion matrix: {str(e)}")
            print(f"Error: {str(e)}")
            return None
    
    def plot_roc_curve(self, fpr: np.ndarray, tpr: np.ndarray, auc_score: float,
                      filename: str = None):
        """Plot ROC curve"""
        
        try:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {auc_score:.4f})')
            ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
            
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc="lower right")
            ax.grid(alpha=0.3)
            plt.tight_layout()
            
            if filename:
                plt.savefig(filename, dpi=100, bbox_inches='tight')
                print(f"[OK] ROC curve saved to {filename}")
            else:
                plt.show()
            
            return fig
        
        except Exception as e:
            logger.error(f"Error plotting ROC curve: {str(e)}")
            print(f"Error: {str(e)}")
            return None


if __name__ == '__main__':
    # Example usage with dummy data
    print("\n" + "="*70)
    print("EVALUATION DEMO WITH SAMPLE DATA")
    print("="*70)
    
    # Create sample classification data
    np.random.seed(42)
    n_samples = 100
    
    # Binary classification
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = np.random.randint(0, 2, n_samples)
    y_pred_proba = np.column_stack([
        1 - np.random.rand(n_samples) * 0.3,
        np.random.rand(n_samples) * 0.7
    ])
    y_pred_proba = y_pred_proba / y_pred_proba.sum(axis=1, keepdims=True)
    
    # Test evaluator
    evaluator = AdvancedEvaluator()
    
    # Evaluate classifier
    metrics = evaluator.evaluate_classifier(
        y_true, y_pred, y_pred_proba,
        class_names=['Low Risk', 'High Risk']
    )
    
    # Plot confusion matrix
    if 'confusion_matrix' in metrics:
        evaluator.plot_confusion_matrix(
            metrics['confusion_matrix'],
            class_names=['Low Risk', 'High Risk'],
            filename=None  # Set to filename to save
        )
    
    # Plot ROC curve
    if 'roc_auc' in metrics:
        evaluator.plot_roc_curve(
            metrics['fpr'],
            metrics['tpr'],
            metrics['roc_auc'],
            filename=None  # Set to filename to save
        )
    
    print("\n[OK] Evaluation complete!")