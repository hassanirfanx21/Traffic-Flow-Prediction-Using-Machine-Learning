"""
Training (Regression) — PyTorch training loop with Huber / MSE Loss
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.config_reg import (DEVICE, BATCH_SIZE, LEARNING_RATE, EPOCHS,
                        PATIENCE, MODEL_DIR)
from src.models_reg import AttentionBiLSTM
import os


def make_dataloaders(train_data, val_data, test_data, batch_size=BATCH_SIZE):
    """Create PyTorch DataLoaders from numpy arrays."""
    def _to_dataset(X, y):
        # Both X and y are floats in regression
        return TensorDataset(
            torch.FloatTensor(X),
            torch.FloatTensor(y)
        )

    train_ds = _to_dataset(*train_data)
    val_ds = _to_dataset(*val_data)
    test_ds = _to_dataset(*test_data)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    print(f"📦 DataLoaders: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}  "
          f"| batch_size={batch_size}")
    return train_loader, val_loader, test_loader


def train_model(model, train_loader, val_loader,
                model_name='model', epochs=EPOCHS, lr=LEARNING_RATE,
                patience=PATIENCE, device=DEVICE):
    """
    Regression training loop using Huber Loss (stable vs outliers).
    """
    model = model.to(device)
    is_attn = isinstance(model, AttentionBiLSTM)

    # Huber loss is excellent for traffic flow — handles spikes better than pure MSE
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': []}
    save_path = os.path.join(MODEL_DIR, f'{model_name}_best.pt')

    print(f"\n🚀 Training {model_name} on {device}  (max {epochs} epochs, patience={patience})")
    print("-" * 70)

    for epoch in range(epochs):
        # ── Train ─────────────────────────────────
        model.train()
        running_loss, total = 0.0, 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()

            if is_attn:
                outputs, _ = model(X_batch)
            else:
                outputs = model(X_batch)

            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * X_batch.size(0)
            total += y_batch.size(0)

        train_loss = running_loss / total

        # ── Validate ──────────────────────────────
        model.eval()
        val_loss, val_total = 0.0, 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                if is_attn:
                    outputs, _ = model(X_batch)
                else:
                    outputs = model(X_batch)
                    
                loss = criterion(outputs, y_batch)
                val_loss += loss.item() * X_batch.size(0)
                val_total += y_batch.size(0)

        val_loss = val_loss / val_total

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        # ── Early stopping ────────────────────────
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or patience_counter >= patience:
            print(f"  Epoch {epoch+1:>3d}/{epochs}  │  "
                  f"Loss (Huber): train={train_loss:.4f} val={val_loss:.4f}  │  "
                  f"LR: {current_lr:.1e}  │  "
                  f"{'✓ best' if patience_counter == 0 else f'patience {patience_counter}/{patience}'}")

        if patience_counter >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch+1}")
            break

    # Load best checkpoint
    model.load_state_dict(torch.load(save_path, weights_only=True))
    print(f"✅ Loaded best checkpoint (val_loss={best_val_loss:.4f})")
    return model, history
