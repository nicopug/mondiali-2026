"""Tier 7 DL: team-embedding + MLP Poisson model.

Symmetric architecture (mirrors XGBoost):
    Input:  team_id, opponent_id, 24 perspective features
    Output: log_lambda (scalar, exp'd to lambda for Poisson)
    Loss:   Poisson NLL = lambda - y * log(lambda)

Per match at inference: 2 forward passes (home-perspective, away-perspective).
Determinism: torch + numpy seeds = 42 (CLAUDE.md §4).
Model serialized via state_dict + JSON config (no pickle, mirroring §5 intent).
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mondiali.config import RANDOM_STATE
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, build_symmetric_rows

log = structlog.get_logger(__name__)

UNK_IDX = 0


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class DLConfig:
    embed_dim: int = 16
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 512
    max_epochs: int = 100
    patience: int = 10
    seed: int = RANDOM_STATE
    grad_clip: float = 1.0


@dataclass
class FeatureStats:
    """Z-score normalization stats computed on training rows."""
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / np.where(self.std > 1e-8, self.std, 1.0)


class PoissonEmbeddingModel(nn.Module):
    def __init__(
        self,
        n_teams: int,
        n_features: int = len(SYMMETRIC_FEATURES),
        embed_dim: int = 16,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.team_emb = nn.Embedding(n_teams + 1, embed_dim, padding_idx=UNK_IDX)
        nn.init.normal_(self.team_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.team_emb.weight[UNK_IDX].zero_()
        self.n_features = n_features
        self.embed_dim = embed_dim
        layers: list[nn.Module] = []
        prev = 2 * embed_dim + n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def forward(
        self, team_ids: torch.Tensor, opp_ids: torch.Tensor, features: torch.Tensor,
    ) -> torch.Tensor:
        et = self.team_emb(team_ids)
        eo = self.team_emb(opp_ids)
        x = torch.cat([et, eo, features], dim=-1)
        h = self.mlp(x)
        return self.head(h).squeeze(-1)


def build_team_index(matches: pd.DataFrame) -> dict[str, int]:
    """Map team names to integer IDs (0 reserved for <UNK>)."""
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return {t: i + 1 for i, t in enumerate(teams)}


def _build_tensors(
    matches: pd.DataFrame,
    team_idx: dict[str, int],
    stats: FeatureStats | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (team_ids, opp_ids, features, targets) tensors for symmetric rows."""
    X, y = build_symmetric_rows(matches, include_tier4=False)
    n = len(matches)
    home_ids = matches["home_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    away_ids = matches["away_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    team_ids = np.empty(2 * n, dtype=np.int64)
    opp_ids = np.empty(2 * n, dtype=np.int64)
    team_ids[0::2] = home_ids
    team_ids[1::2] = away_ids
    opp_ids[0::2] = away_ids
    opp_ids[1::2] = home_ids
    X = np.nan_to_num(X, nan=0.0).astype(np.float64)
    if stats is not None:
        X = stats.transform(X)
    return (
        torch.from_numpy(team_ids),
        torch.from_numpy(opp_ids),
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )


def compute_feature_stats(matches: pd.DataFrame) -> FeatureStats:
    X, _ = build_symmetric_rows(matches, include_tier4=False)
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    return FeatureStats(mean=X.mean(axis=0), std=X.std(axis=0))


def _poisson_nll(log_lambda: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Poisson NLL with log_input=True (matches torch.nn.functional convention)."""
    return F.poisson_nll_loss(log_lambda, y, log_input=True, full=False, reduction="mean")


def train_dl_model(
    train: pd.DataFrame,
    val_es: pd.DataFrame,
    team_idx: dict[str, int],
    config: DLConfig | None = None,
) -> tuple[PoissonEmbeddingModel, FeatureStats, dict]:
    cfg = config or DLConfig()
    _set_seed(cfg.seed)
    device = torch.device("cpu")

    stats = compute_feature_stats(train)
    tr_team, tr_opp, tr_X, tr_y = _build_tensors(train, team_idx, stats)
    va_team, va_opp, va_X, va_y = _build_tensors(val_es, team_idx, stats)

    n_teams = len(team_idx)
    model = PoissonEmbeddingModel(
        n_teams=n_teams,
        n_features=tr_X.shape[1],
        embed_dim=cfg.embed_dim,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    ds = TensorDataset(tr_team, tr_opp, tr_X, tr_y)
    generator = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=generator)

    best_val = float("inf")
    best_state: dict = {}
    patience_left = cfg.patience
    history: list[dict] = []
    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for b_team, b_opp, b_X, b_y in loader:
            opt.zero_grad()
            pred = model(b_team, b_opp, b_X)
            loss = _poisson_nll(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_loss_sum += float(loss.item())
            n_batches += 1
        sched.step()
        train_avg = train_loss_sum / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            val_pred = model(va_team, va_opp, va_X)
            val_loss = float(_poisson_nll(val_pred, va_y).item())
        history.append({"epoch": epoch, "train": train_avg, "val_es": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("early_stopping", epoch=epoch, best_val=best_val)
                break
        if epoch % 5 == 0:
            log.info("epoch", epoch=epoch, train=train_avg, val_es=val_loss)

    if best_state:
        model.load_state_dict(best_state)
    return model, stats, {"best_val_es": best_val, "history": history,
                          "n_epochs_run": len(history)}


def predict_lambda(
    model: PoissonEmbeddingModel,
    matches: pd.DataFrame,
    team_idx: dict[str, int],
    stats: FeatureStats,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict (lam_home, lam_away) for each match."""
    team_ids, opp_ids, X, _ = _build_tensors(matches, team_idx, stats)
    model.eval()
    with torch.no_grad():
        log_lam = model(team_ids, opp_ids, X).numpy()
    lam = np.exp(log_lam)
    return lam[0::2], lam[1::2]


def save_dl_model(
    model: PoissonEmbeddingModel,
    team_idx: dict[str, int],
    stats: FeatureStats,
    config: DLConfig,
    out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "team_idx.json").write_text(json.dumps(team_idx, indent=2))
    (out_dir / "feature_stats.json").write_text(json.dumps({
        "mean": stats.mean.tolist(), "std": stats.std.tolist(),
    }, indent=2))
    cfg_dict = asdict(config)
    cfg_dict["n_teams"] = len(team_idx)
    cfg_dict["n_features"] = model.n_features
    (out_dir / "config.json").write_text(json.dumps(cfg_dict, indent=2))


def load_dl_model(
    in_dir: Path,
) -> tuple[PoissonEmbeddingModel, dict[str, int], FeatureStats, DLConfig]:
    in_dir = Path(in_dir)
    cfg_raw = json.loads((in_dir / "config.json").read_text())
    n_teams = cfg_raw.pop("n_teams")
    n_features = cfg_raw.pop("n_features")
    cfg_raw["hidden_dims"] = tuple(cfg_raw["hidden_dims"])
    cfg = DLConfig(**cfg_raw)
    model = PoissonEmbeddingModel(
        n_teams=n_teams, n_features=n_features,
        embed_dim=cfg.embed_dim, hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )
    state = torch.load(in_dir / "weights.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    team_idx = json.loads((in_dir / "team_idx.json").read_text())
    sj = json.loads((in_dir / "feature_stats.json").read_text())
    stats = FeatureStats(mean=np.array(sj["mean"]), std=np.array(sj["std"]))
    return model, team_idx, stats, cfg
