"""
Master evaluation script for the HAR project.

Runs:
1. Deep learning model evaluation (CNN, LSTM, Hybrid) from saved checkpoints
2. Classical ML evaluation (RF, SVM, XGBoost) with LOSO CV
3. Generates all figures (confusion matrices, model comparison chart)
4. Saves results to JSON for README updates
"""

import sys
import json
import time
import numpy as np
import torch
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config
from src.data.dataset import load_processed_data, get_loso_splits, get_data_loaders, compute_class_weights
from src.training.train import get_model_by_name
from src.training.evaluate import (
    evaluate_model, plot_confusion_matrix, compare_models, get_predictions
)
from src.features.extract import extract_all_features


def evaluate_dl_checkpoint(model_name, X, y, subject_ids, device):
    """Evaluate a deep learning model from its saved checkpoint."""
    checkpoint_path = config.CHECKPOINTS_DIR / f"{model_name}_best.pt"
    if not checkpoint_path.exists():
        print(f"  ⚠️  No checkpoint found for {model_name}, skipping")
        return None

    print(f"\n{'─' * 50}")
    print(f"  Evaluating {model_name.upper()} from checkpoint")
    print(f"{'─' * 50}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    ckpt_cfg = checkpoint.get('config', {}) if isinstance(checkpoint, dict) else {}
    num_channels = int(ckpt_cfg.get('num_channels', X.shape[-1]))
    model = get_model_by_name(model_name, num_channels=num_channels)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Checkpoint fold: {checkpoint.get('fold', '?')}, acc: {checkpoint.get('accuracy', '?')}")

    # Run LOSO evaluation
    fold_results = []
    all_y_true = []
    all_y_pred = []

    for fold, test_subject, train_idx, test_idx in get_loso_splits(subject_ids):
        _, test_loader = get_data_loaders(X, y, train_idx, test_idx)
        metrics = evaluate_model(model, test_loader, device)

        fold_results.append({
            'fold': fold,
            'test_subject': int(test_subject),
            'accuracy': metrics['accuracy'],
            'f1_weighted': metrics['f1_weighted'],
            'f1_macro': metrics['f1_macro'],
        })

        all_y_true.extend(metrics['y_true'].tolist())
        all_y_pred.extend(metrics['y_pred'].tolist())

        print(f"    Fold {fold} (S{test_subject}): "
              f"Acc={metrics['accuracy']:.4f}, F1={metrics['f1_weighted']:.4f}")

    accs = [r['accuracy'] for r in fold_results]
    f1s = [r['f1_weighted'] for r in fold_results]
    f1_macros = [r['f1_macro'] for r in fold_results]

    # Per-class metrics from aggregated predictions
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_score, recall_score, f1_score
    )

    all_labels = list(range(config.NUM_CLASSES))
    target_names = [config.IDX_TO_ACTIVITY.get(i, f'cls_{i}') for i in all_labels]

    cm = confusion_matrix(all_y_true, all_y_pred, labels=all_labels)
    cls_report = classification_report(
        all_y_true, all_y_pred,
        labels=all_labels, target_names=target_names,
        zero_division=0, output_dict=True
    )

    summary = {
        'model': model_name,
        'n_params': n_params,
        'mean_accuracy': float(np.mean(accs)),
        'std_accuracy': float(np.std(accs)),
        'mean_f1_weighted': float(np.mean(f1s)),
        'std_f1_weighted': float(np.std(f1s)),
        'mean_f1_macro': float(np.mean(f1_macros)),
        'std_f1_macro': float(np.std(f1_macros)),
        'fold_results': fold_results,
        'per_class': {name: {
            'precision': cls_report[name]['precision'],
            'recall': cls_report[name]['recall'],
            'f1-score': cls_report[name]['f1-score'],
        } for name in target_names if name in cls_report},
    }

    print(f"\n  {model_name.upper()} Summary:")
    print(f"    Accuracy:     {summary['mean_accuracy']:.4f} ± {summary['std_accuracy']:.4f}")
    print(f"    F1 (weighted): {summary['mean_f1_weighted']:.4f} ± {summary['std_f1_weighted']:.4f}")
    print(f"    F1 (macro):    {summary['mean_f1_macro']:.4f} ± {summary['std_f1_macro']:.4f}")

    # Save confusion matrix figure
    fig_path = config.FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plot_confusion_matrix(cm, title=f"{model_name.upper()} — Confusion Matrix (LOSO)", save_path=fig_path)

    return summary, cm


def run_classical_ml(X, y, subject_ids):
    """Run classical ML models with LOSO CV."""
    print(f"\n{'=' * 60}")
    print("Feature Extraction for Classical ML")
    print(f"{'=' * 60}")

    features = extract_all_features(X)
    print(f"  Feature matrix shape: {features.shape}")

    from src.models.classical import train_and_evaluate

    results = {}
    for model_name in ['rf', 'svm', 'xgboost']:
        try:
            print(f"\n  Training {model_name.upper()}...")
            t0 = time.time()
            res = train_and_evaluate(model_name, features, y, subject_ids, verbose=True)
            elapsed = time.time() - t0
            print(f"  {model_name.upper()} completed in {elapsed:.1f}s")

            # Save confusion matrix
            cm = res['confusion_matrix']
            fig_path = config.FIGURES_DIR / f"confusion_matrix_{model_name}.png"
            plot_confusion_matrix(
                cm, title=f"{model_name.upper()} — Confusion Matrix (LOSO)",
                save_path=fig_path
            )

            results[model_name] = {
                'model': model_name,
                'mean_accuracy': float(res['mean_accuracy']),
                'std_accuracy': float(res['std_accuracy']),
                'mean_f1_weighted': float(res['mean_f1_weighted']),
                'std_f1_weighted': float(res['std_f1_weighted']),
                'mean_f1_macro': float(res['mean_f1_macro']),
                'std_f1_macro': float(res['std_f1_macro']),
                'n_params': 'N/A',
            }
        except Exception as e:
            print(f"  ❌ {model_name} failed: {e}")
            import traceback
            traceback.print_exc()

    return results


def generate_comparison_chart(all_results):
    """Generate model comparison bar chart."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    models = []
    accs = []
    f1s = []
    stds = []

    for name, res in sorted(all_results.items()):
        models.append(name.upper())
        accs.append(res['mean_accuracy'])
        f1s.append(res['mean_f1_weighted'])
        stds.append(res['std_accuracy'])

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(models))
    width = 0.35

    bars1 = ax.bar(x - width/2, accs, width, label='Accuracy',
                   color='#4C72B0', yerr=stds, capsize=4)
    bars2 = ax.bar(x + width/2, f1s, width, label='F1 (weighted)',
                   color='#55A868')

    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison — LOSO Cross-Validation', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 4), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    save_path = config.FIGURES_DIR / "model_comparison.png"
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n  📊 Saved comparison chart to {save_path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("HAR PROJECT — FULL EVALUATION")
    print("=" * 60)

    # Load data
    X, y, subject_ids = load_processed_data()
    print(f"\nData loaded: X={X.shape}, y={y.shape}")
    print(f"Classes: {dict(Counter(y))}")
    print(f"Subjects: {dict(Counter(subject_ids))}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    all_results = {}

    # ─── Deep Learning Models ────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("DEEP LEARNING EVALUATION")
    print(f"{'=' * 60}")

    for model_name in ['cnn', 'lstm', 'hybrid']:
        result = evaluate_dl_checkpoint(model_name, X, y, subject_ids, device)
        if result is not None:
            summary, cm = result
            all_results[model_name] = summary

    # ─── Classical ML Models ─────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("CLASSICAL ML EVALUATION")
    print(f"{'=' * 60}")

    classical_results = run_classical_ml(X, y, subject_ids)
    all_results.update(classical_results)

    # ─── Comparison Chart ────────────────────────────────────────
    generate_comparison_chart(all_results)

    # ─── Save Results ────────────────────────────────────────────
    results_path = config.RESULTS_DIR / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n💾 Results saved to {results_path}")

    # ─── Print Final Summary ─────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("FINAL RESULTS SUMMARY")
    print(f"{'=' * 70}")
    print(f"{'Model':<15} {'Accuracy':>20} {'F1 Weighted':>20} {'F1 Macro':>20}")
    print(f"{'─' * 75}")

    for name in ['rf', 'svm', 'xgboost', 'cnn', 'lstm', 'hybrid']:
        if name in all_results:
            r = all_results[name]
            print(f"{name.upper():<15} "
                  f"{r['mean_accuracy']:.4f} ± {r['std_accuracy']:.4f}   "
                  f"{r['mean_f1_weighted']:.4f} ± {r['std_f1_weighted']:.4f}   "
                  f"{r['mean_f1_macro']:.4f} ± {r['std_f1_macro']:.4f}")

    # Print best model's per-class results
    best_name = max(all_results.keys(),
                    key=lambda k: all_results[k]['mean_accuracy'])
    best = all_results[best_name]

    if 'per_class' in best:
        print(f"\n  Best Model: {best_name.upper()} — Per-Class Results:")
        print(f"  {'Activity':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print(f"  {'─' * 52}")
        for cls_name, metrics in best['per_class'].items():
            print(f"  {cls_name:<20} {metrics['precision']:>10.4f} "
                  f"{metrics['recall']:>10.4f} {metrics['f1-score']:>10.4f}")

    print(f"\n{'=' * 70}")
    print("EVALUATION COMPLETE")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
