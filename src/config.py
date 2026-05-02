"""
Configuration & Hyperparameters for Traffic Congestion Prediction
"""
import os
import torch

# ── Paths ──────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'outputs')
MODEL_DIR = os.path.join(OUTPUT_DIR, 'models')
FIGURE_DIR = os.path.join(OUTPUT_DIR, 'figures')
RESULTS_DIR = os.path.join(OUTPUT_DIR, 'results')
PROCESSED_DIR = os.path.join(OUTPUT_DIR, 'processed')

for d in [MODEL_DIR, FIGURE_DIR, RESULTS_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

# ── Device ─────────────────────────────────────────────────
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Data Parameters ────────────────────────────────────────
TOP_N_STATIONS = 50          # Number of top-traffic stations to use
MIN_PCT_OBSERVED = 50        # Minimum % Observed to consider data reliable
SEQ_LENGTH = 4               # 4 steps × 5 min = 20 minutes lookback
PRED_HORIZON = 1             # Predict 1 step (5 min) ahead

# PeMS column indices (no header in raw files)
RAW_USECOLS = [0, 1, 3, 4, 8, 9, 10, 11]
RAW_COLNAMES = ['Timestamp', 'Station', 'Freeway', 'Direction',
                'Pct_Observed', 'Total_Flow', 'Avg_Occupancy', 'Avg_Speed']

# ── Congestion Thresholds (mph) ────────────────────────────
SPEED_FREE_FLOW = 50         # speed >= 50 → Low congestion
SPEED_MODERATE = 25          # 25 <= speed < 50 → Moderate
                             # speed < 25 → High congestion

# ── Feature Columns (after engineering) ────────────────────
FEATURE_COLS = ['Total_Flow', 'Avg_Occupancy', 'Avg_Speed', 'Pct_Observed',
                'Hour_sin', 'Hour_cos', 'Day_of_week', 'Is_weekend',
                'Flow_Speed_Ratio']
NUM_FEATURES = len(FEATURE_COLS)
NUM_CLASSES = 3
CLASS_NAMES = ['Low', 'Moderate', 'High']

# ── Model Hyperparameters ──────────────────────────────────
HIDDEN_SIZE = 128
NUM_LAYERS = 2
DROPOUT = 0.3
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
EPOCHS = 100
PATIENCE = 10               # Early stopping patience

# ── Train/Val/Test Split (by date strings MM/DD/YYYY) ─────
TRAIN_DATES = ['03/20/2026', '03/21/2026', '03/22/2026', '03/23/2026',
               '03/24/2026', '03/25/2026', '03/26/2026']
VAL_DATES = ['03/27/2026']
TEST_DATES = ['03/28/2026', '03/29/2026']
