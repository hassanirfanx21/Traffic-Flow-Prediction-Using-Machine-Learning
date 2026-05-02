"""
Preprocessing — Feature engineering, congestion labeling, sequence creation, splitting
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.config import (FEATURE_COLS, SEQ_LENGTH, SPEED_FREE_FLOW, SPEED_MODERATE,
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
# 3. CONGESTION LABELING
# ═══════════════════════════════════════════════════════════
def label_congestion(df):
    """Assign congestion labels based on Avg_Speed thresholds."""
    df = df.copy()
    conditions = [
        df['Avg_Speed'] >= SPEED_FREE_FLOW,
        (df['Avg_Speed'] >= SPEED_MODERATE) & (df['Avg_Speed'] < SPEED_FREE_FLOW),
        df['Avg_Speed'] < SPEED_MODERATE
    ]
    labels = [0, 1, 2]  # Low, Moderate, High
    df['Congestion'] = np.select(conditions, labels, default=0)

    counts = df['Congestion'].value_counts().sort_index()
    total = len(df)
    print("\n🏷️  Congestion Label Distribution:")
    for lbl, name in zip([0, 1, 2], ['Low', 'Moderate', 'High']):
        c = counts.get(lbl, 0)
        print(f"   {name:>10s}: {c:>8,}  ({100*c/total:.1f}%)")
    return df


# ═══════════════════════════════════════════════════════════
# 4. SEQUENCE CREATION
# ═══════════════════════════════════════════════════════════
def create_sequences(df, seq_length=SEQ_LENGTH):
    """
    Create sliding-window sequences per station.
    Input:  last `seq_length` intervals' features
    Target: congestion label at the NEXT interval
    Also returns metadata (timestamps, stations) for analysis.
    """
    X_list, y_list, meta_list = [], [], []
    feature_data = df[FEATURE_COLS].values
    label_data = df['Congestion'].values
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
        labs = label_data[idx]
        ts = timestamps[idx]

        for i in range(len(feats) - seq_length):
            X_list.append(feats[i:i + seq_length])
            y_list.append(labs[i + seq_length])  # next step label
            meta_list.append({
                'station': sid,
                'timestamp': ts[i + seq_length],
                'date_str': dates[idx[i + seq_length]] if dates is not None else ''
            })

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int64)
    print(f"\n📦 Created {len(X):,} sequences  |  "
          f"shape: X={X.shape}, y={y.shape}")
    return X, y, meta_list


# ═══════════════════════════════════════════════════════════
# 5. TEMPORAL SPLIT
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
        counts = np.bincount(y[idx], minlength=3)
        print(f"   {name:>5s}: {len(idx):>7,} samples  |  "
              f"Low={counts[0]:,}  Mod={counts[1]:,}  High={counts[2]:,}")
    return splits


# ═══════════════════════════════════════════════════════════
# 6. SCALING
# ═══════════════════════════════════════════════════════════
def scale_features(splits):
    """Fit StandardScaler on train, transform all splits."""
    X_train, y_train = splits['train']
    X_val, y_val = splits['val']
    X_test, y_test = splits['test']

    n_samples, seq_len, n_feat = X_train.shape
    scaler = StandardScaler()

    # Fit on train (flatten → fit → reshape)
    X_train_flat = X_train.reshape(-1, n_feat)
    scaler.fit(X_train_flat)

    X_train_s = scaler.transform(X_train_flat).reshape(n_samples, seq_len, n_feat)
    X_val_s = scaler.transform(X_val.reshape(-1, n_feat)).reshape(X_val.shape)
    X_test_s = scaler.transform(X_test.reshape(-1, n_feat)).reshape(X_test.shape)

    print(f"📏 Scaled features with StandardScaler (fit on train)")
    return (X_train_s, y_train), (X_val_s, y_val), (X_test_s, y_test), scaler


# ═══════════════════════════════════════════════════════════
# 7. FULL PREPROCESSING PIPELINE
# ═══════════════════════════════════════════════════════════
def preprocess_pipeline(df):
    """Run the full preprocessing pipeline."""
    print("=" * 60)
    print("  PREPROCESSING PIPELINE")
    print("=" * 60)

    df = clean_data(df)
    df = engineer_features(df)
    df = label_congestion(df)

    # Add date string for temporal splitting
    df['Date_str'] = df['Timestamp'].dt.strftime('%m/%d/%Y')

    X, y, meta = create_sequences(df)

    print("\n📅 Temporal Train/Val/Test Split:")
    splits = temporal_split(X, y, meta)

    train_data, val_data, test_data, scaler = scale_features(splits)

    return train_data, val_data, test_data, scaler, df
