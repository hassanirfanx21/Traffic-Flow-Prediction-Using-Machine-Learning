"""
Models — Baseline (Gradient Boosting), Standard LSTM, Hybrid Attention-BiLSTM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import NUM_FEATURES, NUM_CLASSES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT


# ═══════════════════════════════════════════════════════════
# MODEL 1: Standard LSTM
# ═══════════════════════════════════════════════════════════
class LSTMModel(nn.Module):
    """
    Standard 2-layer unidirectional LSTM.
    Uses the LAST hidden state for classification.
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0
        )
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, (h_n, _) = self.lstm(x)
        out = h_n[-1]  # last layer's final hidden state
        return self.classifier(out)


# ═══════════════════════════════════════════════════════════
# MODEL 2: Temporal Attention Module
# ═══════════════════════════════════════════════════════════
class TemporalAttention(nn.Module):
    """
    Additive attention over time steps (inspired by HierAttnLSTM paper).
    Learns which past intervals are most important for prediction.

    Input:  (batch, seq_len, hidden_size)
    Output: context vector (batch, hidden_size), attention weights (batch, seq_len)
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.query = nn.Linear(hidden_size, hidden_size // 2)
        self.key = nn.Linear(hidden_size, hidden_size // 2)
        self.energy = nn.Linear(hidden_size // 2, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch, seq_len, hidden_size)
        q = torch.tanh(self.query(lstm_output))       # (batch, seq, H/2)
        k = torch.tanh(self.key(lstm_output))          # (batch, seq, H/2)
        energy = self.energy(q + k).squeeze(-1)        # (batch, seq)
        attn_weights = F.softmax(energy, dim=1)        # (batch, seq)
        context = torch.bmm(
            attn_weights.unsqueeze(1), lstm_output     # (batch, 1, seq) × (batch, seq, H)
        ).squeeze(1)                                    # (batch, H)
        return context, attn_weights


# ═══════════════════════════════════════════════════════════
# MODEL 3: Hybrid Attention-BiLSTM  (OUR CONTRIBUTION)
# ═══════════════════════════════════════════════════════════
class AttentionBiLSTM(nn.Module):
    """
    Hybrid model combining:
    - Bidirectional LSTM (Paper 4: MSTA-GNet idea)
    - Temporal Self-Attention (Paper 3: HierAttnLSTM idea)
    - Missing-data-aware features (Paper 2: LSTM-M idea)
    - Rich temporal encoding (Paper 5: LSTM 2022)

    Architecture:
        Input (batch, seq_len, features)
            → BiLSTM (2 layers, 128 units)
            → Temporal Attention (weighted context)
            → Dense Head (BatchNorm → 256 → 128 → 64 → 3)
            → Softmax → Congestion Class
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS, num_classes=NUM_CLASSES, dropout=DROPOUT):
        super().__init__()
        self.hidden_size = hidden_size

        # Bidirectional LSTM
        self.bilstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )

        # Temporal attention over BiLSTM outputs
        self.attention = TemporalAttention(hidden_size * 2)

        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        # Classification head
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(hidden_size * 2),
            nn.Linear(hidden_size * 2, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: (batch, seq_len, features)
        lstm_out, _ = self.bilstm(x)              # (batch, seq, hidden*2)
        lstm_out = self.layer_norm(lstm_out)        # stabilize
        context, attn_weights = self.attention(lstm_out)   # (batch, hidden*2)
        logits = self.classifier(context)           # (batch, num_classes)
        return logits, attn_weights


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_summary():
    """Print summary of all models."""
    models = {
        'Standard LSTM': LSTMModel(),
        'Attention-BiLSTM': AttentionBiLSTM()
    }
    print(f"{'Model':<25s} {'Parameters':>12s}")
    print("-" * 40)
    for name, model in models.items():
        n = count_parameters(model)
        print(f"{name:<25s} {n:>12,}")
    return models
