"""
Evaluation — Metrics, confusion matrices, visualizations, model comparison
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, f1_score, precision_score,
                             recall_score)
from src.config import CLASS_NAMES, DEVICE, FIGURE_DIR, NUM_CLASSES
from src.models import AttentionBiLSTM
import os


def predict_pytorch(model, data_loader, device=DEVICE):
    """Get predictions and true labels from a PyTorch model."""
    model.eval()
    all_preds, all_labels, all_attn = [], [], []
    is_attn = isinstance(model, AttentionBiLSTM)

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            if is_attn:
                outputs, attn_w = model(X_batch)
                all_attn.append(attn_w.cpu().numpy())
            else:
                outputs = model(X_batch)
            _, predicted = torch.max(outputs, 1)
            all_preds.append(predicted.cpu().numpy())
            all_labels.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    attn = np.concatenate(all_attn) if all_attn else None
    return preds, labels, attn


def evaluate_model(y_true, y_pred, model_name='Model'):
    """Compute and print classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall = recall_score(y_true, y_pred, average='macro', zero_division=0)

    metrics = {
        'model': model_name,
        'accuracy': acc,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Results")
    print(f"{'='*50}")
    print(f"  Accuracy     : {acc:.4f}")
    print(f"  F1 (macro)   : {f1_macro:.4f}")
    print(f"  F1 (weighted): {f1_weighted:.4f}")
    print(f"  Precision    : {precision:.4f}")
    print(f"  Recall       : {recall:.4f}")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0)}")
    return metrics


def plot_confusion_matrices(results, save=True):
    """Plot side-by-side confusion matrices for all models."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        cm = confusion_matrix(res['y_true'], res['y_pred'])
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        ax.set_title(f'{name}\nAccuracy: {res["metrics"]["accuracy"]:.2%}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'confusion_matrices.png'), dpi=150)
    plt.show()


def plot_training_curves(histories, save=True):
    """Plot training/validation loss and accuracy curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        axes[0].plot(epochs, hist['train_loss'], label=f'{name} (train)')
        axes[0].plot(epochs, hist['val_loss'], '--', label=f'{name} (val)')
        
        if 'train_acc' in hist and 'val_acc' in hist:
            axes[1].plot(epochs, hist['train_acc'], label=f'{name} (train)')
            axes[1].plot(epochs, hist['val_acc'], '--', label=f'{name} (val)')

    axes[0].set_title('Loss Curves', fontsize=13, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_title('Accuracy Curves', fontsize=13, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'training_curves.png'), dpi=150)
    plt.show()


def plot_attention_weights(attn_weights, n_samples=8, save=True):
    """Visualize attention weights over time steps."""
    if attn_weights is None:
        print("No attention weights available.")
        return

    fig, axes = plt.subplots(2, 4, figsize=(16, 6))
    indices = np.random.choice(len(attn_weights), min(n_samples, len(attn_weights)), replace=False)

    labels = ['t-20min', 't-15min', 't-10min', 't-5min']
    for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
        weights = attn_weights[idx]
        colors = plt.cm.YlOrRd(weights / weights.max())
        bars = ax.bar(labels, weights, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylim(0, max(0.5, weights.max() * 1.2))
        ax.set_title(f'Sample {idx}', fontsize=10)
        ax.tick_params(axis='x', rotation=45)

    fig.suptitle('Temporal Attention Weights\n(Which past intervals matter most?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'attention_weights.png'), dpi=150)
    plt.show()


def plot_model_comparison(all_metrics, save=True):
    """Bar chart comparing all models across metrics."""
    metrics_to_plot = ['accuracy', 'f1_macro', 'precision', 'recall']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    n_models = len(all_metrics)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, (name, m) in enumerate(all_metrics.items()):
        values = [m[k] for k in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(['Accuracy', 'F1 (Macro)', 'Precision', 'Recall'])
    ax.set_ylabel('Score')
    ax.set_title('Model Comparison on Test Set', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'model_comparison.png'), dpi=150)
    plt.show()
