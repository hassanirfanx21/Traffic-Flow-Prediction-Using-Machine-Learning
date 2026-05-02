"""
Evaluation (Regression) — Metrics (MAE, RMSE, R2), scatter plots, time-series predictions
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.config_reg import DEVICE, FIGURE_DIR, TARGET_COL
from src.models_reg import AttentionBiLSTM
import os


def predict_pytorch(model, data_loader, device=DEVICE):
    """Get continuous predictions and true values from a PyTorch model."""
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
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(y_batch.numpy())

    preds = np.concatenate(all_preds)
    labels = np.concatenate(all_labels)
    attn = np.concatenate(all_attn) if all_attn else None
    return preds, labels, attn


def evaluate_model(y_true, y_pred, y_scaler=None, model_name='Model'):
    """Compute and print regression metrics (MAE, RMSE, R2)."""
    
    # We must inverse transform to compute real-world error metrics
    if y_scaler is not None:
        y_true_real = y_scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()
        y_pred_real = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
    else:
        y_true_real = y_true.flatten()
        y_pred_real = y_pred.flatten()

    mae = mean_absolute_error(y_true_real, y_pred_real)
    mse = mean_squared_error(y_true_real, y_pred_real)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true_real, y_pred_real)

    metrics = {
        'model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2
    }

    print(f"\n{'='*50}")
    print(f"  {model_name} — Test Results on Raw {TARGET_COL}")
    print(f"{'='*50}")
    print(f"  MAE   : {mae:.2f} (avg absolute vehicle prediction error)")
    print(f"  RMSE  : {rmse:.2f} (error heavily penalizing large misses)")
    print(f"  R²    : {r2:.4f} (variance explained)")
    return metrics, y_true_real, y_pred_real


def plot_scatter_predictions(results, save=True):
    """Plot Predicted vs Actual scatter plots for regression."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, res) in zip(axes, results.items()):
        y_true = res['y_true_real']
        y_pred = res['y_pred_real']
        
        # Take a random sample to avoid overplotting if dataset is huge
        sample_size = min(10000, len(y_true))
        idx = np.random.choice(len(y_true), sample_size, replace=False)
        
        ax.scatter(y_true[idx], y_pred[idx], alpha=0.1, s=5, c='#2196F3')
        
        # Perfect prediction line
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([0, max_val], [0, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax.set_title(f'{name}\n$R^2$: {res["metrics"]["R2"]:.3f}',
                     fontsize=13, fontweight='bold')
        ax.set_xlabel(f'Actual {TARGET_COL}')
        ax.set_ylabel(f'Predicted {TARGET_COL}')
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'scatter_predictions_reg.png'), dpi=150)
    plt.show()


def plot_training_curves(histories, save=True):
    """Plot training/validation loss curves."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for name, hist in histories.items():
        epochs = range(1, len(hist['train_loss']) + 1)
        ax.plot(epochs, hist['train_loss'], label=f'{name} (train)')
        ax.plot(epochs, hist['val_loss'], '--', label=f'{name} (val)')

    ax.set_title('Huber Loss Curves (Regression)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (Scaled)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'training_curves_reg.png'), dpi=150)
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

    fig.suptitle('Temporal Attention Weights (Regression)\n(Which past intervals matter most?)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'attention_weights_reg.png'), dpi=150)
    plt.show()


def plot_model_comparison(all_metrics, save=True):
    """Bar chart comparing all models across MAE and RMSE."""
    metrics_to_plot = ['MAE', 'RMSE']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    n_models = len(all_metrics)

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, m) in enumerate(all_metrics.items()):
        values = [m[k] for k in metrics_to_plot]
        offset = (i - n_models / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, alpha=0.85)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f'{val:.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(['MAE (Lower is Better)', 'RMSE (Lower is Better)'])
    ax.set_ylabel('Error Margin (Vehicles)')
    ax.set_title(f'Regression Error Comparison on Test Set ({TARGET_COL})', fontsize=14, fontweight='bold')
    ax.legend()
    # Find max error to set ylim properly
    max_err = max([max([m[k] for k in metrics_to_plot]) for m in all_metrics.values()])
    ax.set_ylim(0, max_err * 1.2)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(FIGURE_DIR, 'model_comparison_reg.png'), dpi=150)
    plt.show()
