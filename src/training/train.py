"""
Training loop with LOSO cross-validation for deep learning HAR models.

Supports:
- CNN1D, LSTM, GRU, HybridCNNLSTM
- Leave-One-Subject-Out cross-validation
- Early stopping with patience
- Learning rate scheduling
- Class-weighted loss for imbalance
- TensorBoard-compatible logging
"""

import sys
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config
from src.data.dataset import (
    load_processed_data, get_loso_splits, get_data_loaders, compute_class_weights
)
from src.training.evaluate import evaluate_model, print_fold_summary


def get_model_by_name(model_name: str, **kwargs):
    """Get a deep learning model by name."""
    if model_name == 'cnn':
        from src.models.cnn import CNN1D
        return CNN1D(**kwargs)
    elif model_name == 'lstm':
        from src.models.rnn import LSTMModel
        return LSTMModel(**kwargs)
    elif model_name == 'gru':
        from src.models.rnn import GRUModel
        return GRUModel(**kwargs)
    elif model_name == 'hybrid':
        from src.models.hybrid import HybridCNNLSTM
        return HybridCNNLSTM(**kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")


class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience=None, min_delta=0.001):
        self.patience = patience or config.EARLY_STOPPING_PATIENCE
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.should_stop


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        logits = model(X_batch)
        loss = criterion(logits, y_batch)
        loss.backward()

        # Gradient clipping for RNNs
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, val_loader, criterion, device):
    """Validate model. Returns average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in val_loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        logits = model(X_batch)
        loss = criterion(logits, y_batch)

        total_loss += loss.item() * X_batch.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += X_batch.size(0)

    return total_loss / total, correct / total


def train_single_fold(model_name, X, y, train_idx, test_idx, fold,
                      test_subject, device, transform=None,
                      num_epochs=None, verbose=True):
    """
    Train a model on a single LOSO fold.

    Returns
    -------
    results : dict with metrics, model state dict, and training history
    """
    num_epochs = num_epochs or config.NUM_EPOCHS

    # Create data loaders
    train_loader, test_loader = get_data_loaders(
        X, y, train_idx, test_idx, transform=transform
    )

    # Initialize model
    model = get_model_by_name(model_name).to(device)

    # Compute class weights from training data
    class_weights = compute_class_weights(y[train_idx]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer and scheduler
    optimizer = Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    early_stopping = EarlyStopping()

    # Training loop
    best_acc = 0.0
    best_state = None
    history = defaultdict(list)

    for epoch in range(num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, test_loader, criterion, device)

        scheduler.step(val_acc)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        elapsed = time.time() - t0

        if verbose and (epoch % 10 == 0 or epoch == num_epochs - 1):
            lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch:3d}/{num_epochs} | "
                  f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
                  f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
                  f"LR={lr:.2e} | {elapsed:.1f}s")

        if early_stopping(val_acc):
            if verbose:
                print(f"  ⏹ Early stopping at epoch {epoch}")
            break

    # Evaluate best model on test set
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate_model(model, test_loader, device)

    results = {
        'fold': fold,
        'test_subject': test_subject,
        'best_val_acc': best_acc,
        'history': dict(history),
        'model_state': best_state,
        **test_metrics,
    }

    return results


def train_loso(model_name: str, num_epochs=None, transform=None,
               verbose=True, save_best=True):
    """
    Full LOSO cross-validation training pipeline.

    Parameters
    ----------
    model_name : str
        One of 'cnn', 'lstm', 'gru', 'hybrid'.
    num_epochs : int, optional
    transform : callable, optional
        Data augmentation function.
    verbose : bool
    save_best : bool
        Whether to save the best model checkpoint.

    Returns
    -------
    all_results : dict with per-fold and aggregate metrics
    """
    # Load data
    X, y, subject_ids = load_processed_data()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'=' * 60}")
    print(f"Training {model_name.upper()} with LOSO CV on {device}")
    print(f"Data shape: X={X.shape}, y={y.shape}")
    print(f"{'=' * 60}")

    fold_results = []

    for fold, test_subject, train_idx, test_idx in get_loso_splits(subject_ids):
        print(f"\n  ── Fold {fold} (Test Subject: {test_subject}) ──")
        print(f"     Train: {len(train_idx)}, Test: {len(test_idx)}")

        results = train_single_fold(
            model_name, X, y, train_idx, test_idx,
            fold, test_subject, device,
            transform=transform,
            num_epochs=num_epochs,
            verbose=verbose,
        )

        fold_results.append(results)
        print_fold_summary(results)

    # Aggregate results
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_weighted'] for r in fold_results]
    f1_macros = [r['f1_macro'] for r in fold_results]

    summary = {
        'model': model_name,
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accs),
        'std_accuracy': np.std(accs),
        'mean_f1_weighted': np.mean(f1s),
        'std_f1_weighted': np.std(f1s),
        'mean_f1_macro': np.mean(f1_macros),
        'std_f1_macro': np.std(f1_macros),
    }

    print(f"\n{'=' * 60}")
    print(f"{model_name.upper()} LOSO Summary:")
    print(f"  Accuracy:     {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"  F1 (weighted): {summary['mean_f1_weighted']:.4f} ± {summary['std_f1_weighted']:.4f}")
    print(f"  F1 (macro):    {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")
    print(f"{'=' * 60}")

    # Save best fold's model
    if save_best:
        best_fold = max(fold_results, key=lambda r: r['accuracy'])
        save_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
        torch.save({
            'model_name': model_name,
            'model_state_dict': best_fold['model_state'],
            'fold': best_fold['fold'],
            'accuracy': best_fold['accuracy'],
            'config': {
                'num_channels': config.NUM_CHANNELS,
                'num_classes': config.NUM_CLASSES,
                'window_size': config.WINDOW_SIZE,
            },
        }, save_path)
        print(f"\n💾 Best model saved to {save_path}")

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train HAR models with LOSO CV")
    parser.add_argument('--model', type=str, default='cnn',
                        choices=['cnn', 'lstm', 'gru', 'hybrid'],
                        help='Model architecture')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')
    parser.add_argument('--folds', type=int, default=None,
                        help='Limit number of folds (for quick testing)')

    args = parser.parse_args()
    train_loso(args.model, num_epochs=args.epochs)
