"""
Classical machine learning models for HAR.

Implements Random Forest, SVM, and XGBoost with
Leave-One-Subject-Out cross-validation support.
"""

import sys
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import config


def get_model(name: str, random_state: int = None):
    """
    Get a classical ML model by name.

    Parameters
    ----------
    name : str
        One of 'rf', 'svm', 'xgboost'.
    random_state : int, optional

    Returns
    -------
    sklearn Pipeline with StandardScaler + classifier
    """
    rs = random_state or config.RANDOM_SEED

    if name == 'rf':
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=5,
            min_samples_leaf=2,
            n_jobs=-1,
            random_state=rs,
            class_weight='balanced',
        )
    elif name == 'svm':
        clf = SVC(
            kernel='rbf',
            C=10.0,
            gamma='scale',
            class_weight='balanced',
            random_state=rs,
        )
    elif name == 'xgboost':
        try:
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric='mlogloss',
                n_jobs=-1,
                random_state=rs,
            )
        except ImportError:
            raise ImportError("xgboost not installed. Run: pip install xgboost")
    else:
        raise ValueError(f"Unknown model: {name}. Choose from 'rf', 'svm', 'xgboost'.")

    # Wrap in pipeline with scaling
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', clf),
    ])

    return pipeline


def train_and_evaluate(model_name: str, X_features: np.ndarray,
                       y: np.ndarray, subject_ids: np.ndarray,
                       verbose: bool = True) -> dict:
    """
    Train and evaluate a classical ML model using LOSO CV.

    Parameters
    ----------
    model_name : str
        One of 'rf', 'svm', 'xgboost'.
    X_features : np.ndarray, shape (N, num_features)
        Extracted feature vectors.
    y : np.ndarray, shape (N,)
        Class labels.
    subject_ids : np.ndarray, shape (N,)
        Subject IDs for LOSO splits.

    Returns
    -------
    results : dict with per-fold and aggregate metrics
    """
    from src.data.dataset import get_loso_splits

    unique_subjects = np.unique(subject_ids)
    fold_results = []

    if verbose:
        print(f"\n{'=' * 60}")
        print(f"Training {model_name.upper()} with LOSO CV ({len(unique_subjects)} folds)")
        print(f"{'=' * 60}")

    all_y_true = []
    all_y_pred = []

    for fold, test_subject, train_idx, test_idx in get_loso_splits(subject_ids):
        X_train, X_test = X_features[train_idx], X_features[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Handle NaN/Inf in features
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        # Train
        model = get_model(model_name)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        all_labels = list(range(config.NUM_CLASSES))
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0, labels=all_labels)
        f1_macro = f1_score(y_test, y_pred, average='macro', zero_division=0, labels=all_labels)

        fold_results.append({
            'fold': fold,
            'test_subject': test_subject,
            'accuracy': acc,
            'f1_weighted': f1,
            'f1_macro': f1_macro,
            'n_train': len(train_idx),
            'n_test': len(test_idx),
        })

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())

        if verbose:
            print(f"  Fold {fold} (Subject {test_subject}): "
                  f"Acc={acc:.4f}, F1={f1:.4f} (macro={f1_macro:.4f})")

    # Aggregate results
    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_weighted'] for r in fold_results]
    f1_macros = [r['f1_macro'] for r in fold_results]

    results = {
        'model': model_name,
        'fold_results': fold_results,
        'mean_accuracy': np.mean(accs),
        'std_accuracy': np.std(accs),
        'mean_f1_weighted': np.mean(f1s),
        'std_f1_weighted': np.std(f1s),
        'mean_f1_macro': np.mean(f1_macros),
        'std_f1_macro': np.std(f1_macros),
        'confusion_matrix': confusion_matrix(all_y_true, all_y_pred, labels=list(range(config.NUM_CLASSES))),
        'classification_report': classification_report(
            all_y_true, all_y_pred,
            labels=list(range(config.NUM_CLASSES)),
            target_names=[config.IDX_TO_ACTIVITY[i] for i in range(config.NUM_CLASSES)],
            zero_division=0,
        ),
    }

    if verbose:
        print(f"\n  {'─' * 40}")
        print(f"  {model_name.upper()} LOSO Results:")
        print(f"    Accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        print(f"    F1 (weighted): {results['mean_f1_weighted']:.4f} ± {results['std_f1_weighted']:.4f}")
        print(f"    F1 (macro): {results['mean_f1_macro']:.4f} ± {results['std_f1_macro']:.4f}")
        print(f"\n  Classification Report:\n{results['classification_report']}")

    return results


def compare_all_models(X_features: np.ndarray, y: np.ndarray,
                       subject_ids: np.ndarray) -> dict:
    """Train and compare all classical ML models."""
    all_results = {}
    for name in ['rf', 'svm', 'xgboost']:
        try:
            all_results[name] = train_and_evaluate(
                name, X_features, y, subject_ids
            )
        except Exception as e:
            print(f"  ❌ {name} failed: {e}")

    # Print comparison table
    print(f"\n{'=' * 60}")
    print("Model Comparison (LOSO CV)")
    print(f"{'=' * 60}")
    print(f"{'Model':<12} {'Accuracy':>12} {'F1 (weighted)':>15} {'F1 (macro)':>12}")
    print(f"{'─' * 55}")

    for name, res in all_results.items():
        print(f"{name.upper():<12} "
              f"{res['mean_accuracy']:.4f}±{res['std_accuracy']:.4f}  "
              f"{res['mean_f1_weighted']:.4f}±{res['std_f1_weighted']:.4f}  "
              f"{res['mean_f1_macro']:.4f}±{res['std_f1_macro']:.4f}")

    return all_results
