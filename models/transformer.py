"""Tiny temporal Transformer for educational demonstration."""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    torch = None
    nn = None

from config import (
    SEED, LOOKBACK, HORIZON, EXOG_FEATURES,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD, TRANSFORMER_DIM_FEEDFORWARD,
    TRANSFORMER_DROPOUT, TRANSFORMER_NUM_LAYERS, TRANSFORMER_LR,
    TRANSFORMER_BATCH_SIZE, TRANSFORMER_EPOCHS, TRANSFORMER_GRAD_CLIP,
    QUANTILE_LOW, QUANTILE_HIGH
)
from models.base import ForecastModel


class TinyTemporalTransformer(nn.Module):
    """Minimal Transformer encoder for sequence-to-sequence forecasting."""

    def __init__(
        self,
        past_dim: int,
        future_dim: int,
        d_model: int = TRANSFORMER_D_MODEL,
        nhead: int = TRANSFORMER_NHEAD,
        dim_feedforward: int = TRANSFORMER_DIM_FEEDFORWARD,
        dropout: float = TRANSFORMER_DROPOUT,
        num_layers: int = TRANSFORMER_NUM_LAYERS,
        lookback: int = LOOKBACK,
    ):
        super().__init__()
        self.proj = nn.Linear(past_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True, activation="gelu"
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.future_proj = nn.Linear(future_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, 32), nn.GELU(), nn.Linear(32, 1)
        )
        self.position = nn.Parameter(torch.zeros(1, lookback, d_model))
        nn.init.normal_(self.position, mean=0.0, std=0.02)

    def forward(self, x_past: torch.Tensor, x_future: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(self.proj(x_past) + self.position)
        context = encoded[:, -1, :].unsqueeze(1).expand(-1, x_future.shape[1], -1)
        future_emb = self.future_proj(x_future)
        return self.head(torch.cat([context, future_emb], dim=-1)).squeeze(-1)


class TransformerModel(ForecastModel):
    """Wrapper for TinyTemporalTransformer with preprocessing and training."""

    def __init__(self, epochs: int = TRANSFORMER_EPOCHS):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is not installed. Install requirements-transformer.txt to enable this mode.")
        self._epochs = epochs
        self._pack: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return "Tiny temporal Transformer (CPU)"

    def fit(self, df: pd.DataFrame, target: str) -> "TransformerModel":
        """Train Transformer on historical data with validation."""
        torch.set_num_threads(min(2, max(1, torch.get_num_threads())))
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        split = len(df) - 7 * 24
        target_mean = float(df.iloc[:split][target].mean())
        target_std = float(df.iloc[:split][target].std() + 1e-6)
        exog_mean = df.iloc[:split][EXOG_FEATURES].mean().to_numpy(dtype=float)
        exog_std = (df.iloc[:split][EXOG_FEATURES].std().to_numpy(dtype=float) + 1e-6)

        exog_raw = df[EXOG_FEATURES].to_numpy(dtype=float)
        exog_z = (exog_raw - exog_mean) / exog_std
        y_raw = df[target].to_numpy(dtype=float)
        y_z = (y_raw - target_mean) / target_std

        past = np.column_stack([y_z, exog_z]).astype(np.float32)
        future_cov = exog_z.astype(np.float32)

        def windows(starts: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            xp, xf, yy = [], [], []
            for t in starts:
                xp.append(past[t - LOOKBACK:t])
                xf.append(future_cov[t:t + HORIZON])
                yy.append(y_z[t:t + HORIZON])
            if not xp:
                return (
                    np.empty((0, LOOKBACK, past.shape[1]), np.float32),
                    np.empty((0, HORIZON, future_cov.shape[1]), np.float32),
                    np.empty((0, HORIZON), np.float32)
                )
            return np.asarray(xp, np.float32), np.asarray(xf, np.float32), np.asarray(yy, np.float32)

        train_starts = list(range(LOOKBACK, split - HORIZON + 1, 2))
        val_starts = list(range(split, len(df) - HORIZON + 1, 3))

        xpt, xft, yt = windows(train_starts)
        xpv, xfv, yv = windows(val_starts)

        device = torch.device("cpu")
        model = TinyTemporalTransformer(past.shape[1], future_cov.shape[1]).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=TRANSFORMER_LR)
        loss_fn = nn.MSELoss()

        ds = TensorDataset(torch.from_numpy(xpt), torch.from_numpy(xft), torch.from_numpy(yt))
        loader = DataLoader(ds, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)

        model.train()
        for _ in range(self._epochs):
            for bx, bf, by in loader:
                opt.zero_grad(set_to_none=True)
                pred = model(bx, bf)
                loss = loss_fn(pred, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRANSFORMER_GRAD_CLIP)
                opt.step()

        # Validation metrics
        model.eval()
        with torch.no_grad():
            val_pred_z = model(torch.from_numpy(xpv), torch.from_numpy(xfv)).cpu().numpy()
        val_pred = val_pred_z * target_std + target_mean
        val_actual = yv * target_std + target_mean
        resid = (val_actual - val_pred).reshape(-1)
        q_low, q_high = np.quantile(resid, [QUANTILE_LOW, QUANTILE_HIGH]).tolist() if len(resid) else (-8.0, 8.0)
        metrics = model_metrics(val_actual.reshape(-1), val_pred.reshape(-1))

        # Fine-tune on all data for forward forecast
        all_starts = list(range(LOOKBACK, len(df) - HORIZON + 1, 2))
        xpa, xfa, ya = windows(all_starts)
        ds_all = TensorDataset(torch.from_numpy(xpa), torch.from_numpy(xfa), torch.from_numpy(ya))
        loader_all = DataLoader(ds_all, batch_size=TRANSFORMER_BATCH_SIZE, shuffle=True)

        model.train()
        for _ in range(max(2, self._epochs // 2)):
            for bx, bf, by in loader_all:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(model(bx, bf), by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), TRANSFORMER_GRAD_CLIP)
                opt.step()

        model.eval()
        self._pack = {
            "model": model,
            "target_mean": target_mean,
            "target_std": target_std,
            "exog_mean": exog_mean,
            "exog_std": exog_std,
            "lookback": LOOKBACK,
            "q_low": float(q_low),
            "q_high": float(q_high),
            "metrics": metrics,
            "model_type": "Tiny temporal Transformer (CPU)",
        }
        return self

    def predict(self, history: pd.DataFrame, target: str, future: pd.DataFrame) -> np.ndarray:
        """Generate forecast using trained Transformer."""
        if self._pack is None:
            raise RuntimeError("Model must be fitted before prediction.")

        exog_hist = (history[EXOG_FEATURES].to_numpy(dtype=float) - self._pack["exog_mean"]) / self._pack["exog_std"]
        y_hist = (history[target].to_numpy(dtype=float) - self._pack["target_mean"]) / self._pack["target_std"]
        past = np.column_stack([y_hist, exog_hist]).astype(np.float32)[-self._pack["lookback"]:]
        future_exog = ((future[EXOG_FEATURES].to_numpy(dtype=float) - self._pack["exog_mean"]) / self._pack["exog_std"]).astype(np.float32)

        with torch.no_grad():
            pred_z = self._pack["model"](
                torch.from_numpy(past[None, :, :]),
                torch.from_numpy(future_exog[None, :, :])
            ).cpu().numpy()[0]

        return pred_z * self._pack["target_std"] + self._pack["target_mean"]

    def backtest(self, df: pd.DataFrame, target: str, split: int) -> dict[str, Any]:
        """Return validation metrics from training."""
        if self._pack is None:
            raise RuntimeError("Model must be fitted before backtest.")
        return {
            "metrics": self._pack["metrics"],
            "interval": (self._pack["q_low"], self._pack["q_high"]),
        }

    @property
    def interval(self) -> tuple[float, float]:
        if self._pack is None:
            return (-8.0, 8.0)
        return (self._pack["q_low"], self._pack["q_high"])

    @property
    def pack(self) -> dict[str, Any] | None:
        return self._pack


def model_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Compute MAE and RMSE."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    return {
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }
