"""
Evaluation utilities for HAR models.

Provides:
- Per-class and aggregate metrics (accuracy, F1, precision, recall)
- Confusion matrix computation and visualization
- LOSO fold summary printing
"""

import sys
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


@torch.no_grad()
def get_predictions(model, data_loader, device):
    """
    Get model predictions on an entire dataset.

    Returns
    -------
    y_true : np.ndarray
    y_pred : np.ndarray
    y_probs : np.ndarray, shape (N, num_classes)
    """
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    for X_batch, y_batch in data_loader:
        X_batch = X_batch.to(device)
        logits = model(X_batch)
        probs = torch.softmax(logits, dim=1)

        all_true.append(y_batch.numpy())
        all_pred.append(logits.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_probs),
    )


def evaluate_model(model, data_loader, device) -> dict:
    """
    Evaluate a model and compute all metrics.

    Returns
    -------
    metrics : dict with accuracy, f1, precision, recall, confusion matrix, etc.
    """
    y_true, y_pred, y_probs = get_predictions(model, data_loader, device)

    all_labels = list(range(config.NUM_CLASSES))
    target_names = [config.IDX_TO_ACTIVITY.get(i, f'cls_{i}') for i in all_labels]

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0, labels=all_labels),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0, labels=all_labels),
        'confusion_matrix': confusion_matrix(y_true, y_pred, labels=all_labels),
        'y_true': y_true,
        'y_pred': y_pred,
        'y_probs': y_probs,
        'classification_report': classification_report(
            y_true, y_pred,
            labels=all_labels,
            target_names=target_names,
            zero_division=0,
        ),
    }

    return metrics


def print_fold_summary(results: dict):
    """Print a concise summary for a single LOSO fold."""
    print(f"     ✅ Fold {results['fold']} (Subject {results['test_subject']}): "
          f"Acc={results['accuracy']:.4f}, "
          f"F1={results['f1_weighted']:.4f} "
          f"(macro={results['f1_macro']:.4f})")


def plot_confusion_matrix(cm, title="Confusion Matrix", save_path=None,
                          normalize=True):
    """
    Plot a confusion matrix heatmap.

    Parameters
    ----------
    cm : np.ndarray, shape (num_classes, num_classes)
    title : str
    save_path : str or Path, optional
    normalize : bool
        If True, show percentages instead of counts.
    """
    class_names = [config.IDX_TO_ACTIVITY.get(i, f'{i}') for i in range(cm.shape[0])]

    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = '.2f'
    else:
        cm_display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_display,
        annot=True, fmt=fmt,
        xticklabels=class_names,
        yticklabels=class_names,
        cmap='Blues',
        square=True,
        ax=ax,
    )
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    ax.set_title(title, fontsize=14)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📊 Saved confusion matrix to {save_path}")

    plt.close(fig)
    return fig


def plot_training_history(history: dict, title="Training History",
                          save_path=None):
    """
    Plot training and validation loss/accuracy curves.

    Parameters
    ----------
    history : dict with keys 'train_loss', 'val_loss', 'train_acc', 'val_acc'
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{title} — Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title(f'{title} — Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  📈 Saved training history to {save_path}")

    plt.close(fig)
    return fig


def compare_models(results_dict: dict, save_path=None):
    """
    Create a comparison table and bar chart of multiple models.

    Parameters
    ----------
    results_dict : dict of {model_name: summary_dict}
    """
    print(f"\n{'=' * 70}")
    print("Model Comparison (LOSO CV)")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Accuracy':>15} {'F1 (weighted)':>17} {'F1 (macro)':>15}")
    print(f"{'─' * 65}")

    models = []
    accs = []
    f1s = []

    for name, res in results_dict.items():
        print(f"{name.upper():<15} "
              f"{res['mean_accuracy']:.4f} ± {res['std_accuracy']:.4f}   "
              f"{res['mean_f1_weighted']:.4f} ± {res['std_f1_weighted']:.4f}   "
              f"{res['mean_f1_macro']:.4f} ± {res['std_f1_macro']:.4f}")
        models.append(name.upper())
        accs.append(res['mean_accuracy'])
        f1s.append(res['mean_f1_weighted'])

    # Bar chart
    if save_path:
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(models))
        width = 0.35

        bars1 = ax.bar(x - width/2, accs, width, label='Accuracy', color='#4C72B0')
        bars2 = ax.bar(x + width/2, f1s, width, label='F1 (weighted)', color='#55A868')

        ax.set_ylabel('Score')
        ax.set_title('Model Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(models)
        ax.legend()
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"\n  📊 Saved comparison chart to {save_path}")
        plt.close(fig)
