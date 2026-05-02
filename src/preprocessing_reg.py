"""
Preprocessing (Regression) — Feature engineering, sequence creation, scaling
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config_reg import (FEATURE_COLS, SEQ_LENGTH, TARGET_COL,
                        TRAIN_DATES, VAL_DATES, TEST_DATES, PROCESSED_DIR)
import os


# ═══════════════════════════════════════════════════════════
# 1. CLEANING
# ═══════════════════════════════════════════════════════════
def clean_data(df):
    """Handle missing values and outliers."""
    df = df.copy()
    n_before = len(df)

    # Drop rows where core features are all NaN
    df.dropna(subset=['Total_Flow', 'Avg_Speed'], how='all', inplace=True)

    # Fill remaining NaNs
    df['Total_Flow'] = df['Total_Flow'].fillna(0)
    df['Avg_Occupancy'] = df['Avg_Occupancy'].fillna(0)
    df['Pct_Observed'] = df['Pct_Observed'].fillna(0)

    # For speed: forward-fill within each station, then fill with median
    df['Avg_Speed'] = df.groupby('Station')['Avg_Speed'].transform(
        lambda s: s.ffill().bfill()
    )
    median_speed = df['Avg_Speed'].median()
    df['Avg_Speed'] = df['Avg_Speed'].fillna(median_speed)

    # Clip outliers
    df['Total_Flow'] = df['Total_Flow'].clip(lower=0, upper=3000)
    df['Avg_Speed'] = df['Avg_Speed'].clip(lower=0, upper=100)
    df['Avg_Occupancy'] = df['Avg_Occupancy'].clip(lower=0, upper=1)

    print(f"🧹 Cleaned: {n_before:,} → {len(df):,} rows  "
          f"(dropped {n_before - len(df):,})")
    return df


# ═══════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════
def engineer_features(df):
    """Add temporal and interaction features."""
    df = df.copy()

    # Temporal features
    df['Hour'] = df['Timestamp'].dt.hour + df['Timestamp'].dt.minute / 60.0
    df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
    df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
    df['Day_of_week'] = df['Timestamp'].dt.dayofweek          # 0=Mon … 6=Sun
    df['Is_weekend'] = (df['Day_of_week'] >= 5).astype(float)

    # Congestion proxy: flow-to-speed ratio
    df['Flow_Speed_Ratio'] = df['Total_Flow'] / (df['Avg_Speed'] + 1)

    # Normalize Pct_Observed to 0-1
    df['Pct_Observed'] = df['Pct_Observed'] / 100.0

    print(f"🔧 Engineered {len(FEATURE_COLS)} features: {FEATURE_COLS}")
    return df


# ═══════════════════════════════════════════════════════════
# 3. SEQUENCE CREATION
# ═══════════════════════════════════════════════════════════
def create_sequences(df, seq_length=SEQ_LENGTH):
    """
    Create sliding-window sequences per station.
    Input:  last `seq_length` intervals' features
    Target: continuous TARGET_COL at the NEXT interval
    """
    X_list, y_list, meta_list = [], [], []
    feature_data = df[FEATURE_COLS].values
    target_data = df[TARGET_COL].values
    timestamps = df['Timestamp'].values
    stations = df['Station'].values
    dates = df['Date_str'].values if 'Date_str' in df.columns else None

    # Group indices by station for correct slicing
    station_ids = df['Station'].values
    unique_stations = df['Station'].unique()

    for sid in unique_stations:
        mask = station_ids == sid
        idx = np.where(mask)[0]

        feats = feature_data[idx]
        targs = target_data[idx]
        ts = timestamps[idx]

        for i in range(len(feats) - seq_length):
            X_list.append(feats[i:i + seq_length])
            y_list.append(targs[i + seq_length])  # next step target
            meta_list.append({
                'station': sid,
                'timestamp': ts[i + seq_length],
                'date_str': dates[idx[i + seq_length]] if dates is not None else ''
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32).reshape(-1, 1)  # Reshape for scaling
    print(f"\n📦 Created {len(X):,} sequences  |  "
          f"shape: X={X.shape}, y={y.shape}")
    return X, y, meta_list


# ═══════════════════════════════════════════════════════════
# 4. TEMPORAL SPLIT
# ═══════════════════════════════════════════════════════════
def temporal_split(X, y, meta):
    """Split by date — no data leakage from future to past."""
    train_set = set(TRAIN_DATES)
    val_set = set(VAL_DATES)
    test_set = set(TEST_DATES)

    train_idx, val_idx, test_idx = [], [], []
    for i, m in enumerate(meta):
        d = m['date_str']
        if d in train_set:
            train_idx.append(i)
        elif d in val_set:
            val_idx.append(i)
        elif d in test_set:
            test_idx.append(i)

    splits = {}
    for name, idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        idx = np.array(idx)
        splits[name] = (X[idx], y[idx])
        print(f"   {name:>5s}: {len(idx):>7,} samples")
    return splits


# ═══════════════════════════════════════════════════════════
# 5. SCALING (Features + Target)
# ═══════════════════════════════════════════════════════════
def scale_features(splits):
    """Fit StandardScaler on train features AND target, transform all splits."""
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    n_samples, seq_len, n_feat = X_train.shape
    
    # Scale Features (X)
    X_scaler = StandardScaler()
    X_train_flat = X_train.reshape(-1, n_feat)
    X_scaler.fit(X_train_flat)
    
    X_train_s = X_scaler.transform(X_train_flat).reshape(n_samples, seq_len, n_feat)
    X_val_s = X_scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test_s = X_scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)
    
    # Scale Target (y) - Critical for Regression Stability
    y_scaler = StandardScaler()
    y_train_s = y_scaler.fit_transform(y_train)
    y_val_s = y_scaler.transform(y_val)
    y_test_s = y_scaler.transform(y_test)

    print(f"📏 Scaled features and target with StandardScalers (fit on train)")
    return (X_train_s, y_train_s), (X_val_s, y_val_s), (X_test_s, y_test_s), X_scaler, y_scaler


# ═══════════════════════════════════════════════════════════
# 6. FULL PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════
def preprocess_pipeline(df):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  PREPROCESSING PIPELINE (REGRESSION)")
    print("=" * 60)

    df = clean_data(df)
    df = engineer_features(df)

    # Add date string for temporal splitting
    df['Date_str'] = df['Timestamp'].dt.strftime('%m/%d/%Y')

    X, y, meta = create_sequences(df)

    print("\n📅 Temporal Train/Val/Test Split:")
    splits = temporal_split(X, y, meta)

    train_data, val_data, test_data, X_scaler, y_scaler = scale_features(splits)

    return train_data, val_data, test_data, X_scaler, y_scaler, df
