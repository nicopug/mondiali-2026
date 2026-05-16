"""Tier 7-L3: bivariate Poisson with per-match learned correlation (rho).

Non-symmetric (NN takes home/away in fixed roles) → predicts
(log_lambda_home, log_lambda_away, rho_logit). Rho constrained to (-0.3, 0.3)
via tanh. Loss = negative log Dixon-Coles-corrected joint probability at the
observed score cell (independence + per-match small DC correction).

For symmetry at inference, the model is trained with BOTH natural and swapped
orientation pairs (data augmentation), so it behaves consistently when
home/away are swapped at neutral venue.
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
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mondiali.config import RANDOM_STATE
from mondiali.model.poisson_xgb import SYMMETRIC_FEATURES, build_symmetric_rows

log = structlog.get_logger(__name__)

UNK_IDX = 0
RHO_BOUND = 0.3


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class BivariateConfig:
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
    mean: np.ndarray
    std: np.ndarray

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / np.where(self.std > 1e-8, self.std, 1.0)


def build_team_index(matches: pd.DataFrame) -> dict[str, int]:
    teams = sorted(set(matches["home_team"]) | set(matches["away_team"]))
    return {t: i + 1 for i, t in enumerate(teams)}


def _natural_features(matches: pd.DataFrame, stats: FeatureStats | None = None) -> np.ndarray:
    """Use just the home-perspective rows from build_symmetric_rows (no doubling)."""
    X, _ = build_symmetric_rows(matches, include_tier4=False)
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    X_home = X[0::2]  # home-perspective only — 1 row per match
    if stats is not None:
        X_home = stats.transform(X_home)
    return X_home.astype(np.float32)


def _swapped_features(matches: pd.DataFrame, stats: FeatureStats | None = None) -> np.ndarray:
    X, _ = build_symmetric_rows(matches, include_tier4=False)
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    X_away = X[1::2]
    if stats is not None:
        X_away = stats.transform(X_away)
    return X_away.astype(np.float32)


class BivariatePoissonModel(nn.Module):
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
        self.trunk = nn.Sequential(*layers)
        # 3 heads: log_lambda_home, log_lambda_away, rho_logit
        self.lh_head = nn.Linear(prev, 1)
        self.la_head = nn.Linear(prev, 1)
        self.rho_head = nn.Linear(prev, 1)

    def forward(
        self, home_ids: torch.Tensor, away_ids: torch.Tensor, features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        eh = self.team_emb(home_ids)
        ea = self.team_emb(away_ids)
        x = torch.cat([eh, ea, features], dim=-1)
        h = self.trunk(x)
        log_lh = self.lh_head(h).squeeze(-1)
        log_la = self.la_head(h).squeeze(-1)
        rho = torch.tanh(self.rho_head(h).squeeze(-1)) * RHO_BOUND
        return log_lh, log_la, rho


def dc_log_prob_at_cell(
    log_lh: torch.Tensor, log_la: torch.Tensor, rho: torch.Tensor,
    h: torch.Tensor, a: torch.Tensor,
) -> torch.Tensor:
    """log P(h, a) under DC-corrected independent Poisson. NOT renormalized."""
    lh = torch.exp(log_lh)
    la = torch.exp(log_la)
    log_p_h = -lh + h.float() * log_lh - torch.lgamma(h.float() + 1)
    log_p_a = -la + a.float() * log_la - torch.lgamma(a.float() + 1)
    log_base = log_p_h + log_p_a
    eps = 1e-6
    tau_00 = torch.log(torch.clamp(1.0 - lh * la * rho, min=eps))
    tau_01 = torch.log(torch.clamp(1.0 + lh * rho, min=eps))
    tau_10 = torch.log(torch.clamp(1.0 + la * rho, min=eps))
    tau_11 = torch.log(torch.clamp(1.0 - rho, min=eps))
    mask_00 = ((h == 0) & (a == 0)).float()
    mask_01 = ((h == 0) & (a == 1)).float()
    mask_10 = ((h == 1) & (a == 0)).float()
    mask_11 = ((h == 1) & (a == 1)).float()
    log_tau = mask_00 * tau_00 + mask_01 * tau_01 + mask_10 * tau_10 + mask_11 * tau_11
    return log_base + log_tau


def _build_tensors(
    matches: pd.DataFrame, team_idx: dict[str, int], stats: FeatureStats | None,
) -> tuple[torch.Tensor, ...]:
    """Return home_ids, away_ids, features, h_goals, a_goals (1 row/match)."""
    home_ids = matches["home_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    away_ids = matches["away_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    X = _natural_features(matches, stats=stats)
    h = matches["home_score"].astype(int).to_numpy()
    a = matches["away_score"].astype(int).to_numpy()
    return (
        torch.from_numpy(home_ids.astype(np.int64)),
        torch.from_numpy(away_ids.astype(np.int64)),
        torch.from_numpy(X),
        torch.from_numpy(h.astype(np.int64)),
        torch.from_numpy(a.astype(np.int64)),
    )


def _build_tensors_with_aug(
    matches: pd.DataFrame, team_idx: dict[str, int], stats: FeatureStats,
) -> tuple[torch.Tensor, ...]:
    """Natural + swapped orientation rows for symmetric training."""
    home_ids = matches["home_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    away_ids = matches["away_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    Xnat = _natural_features(matches, stats=stats)
    Xswap = _swapped_features(matches, stats=stats)
    h = matches["home_score"].astype(int).to_numpy()
    a = matches["away_score"].astype(int).to_numpy()
    # Concat natural + swapped
    home_ids_all = np.concatenate([home_ids, away_ids])
    away_ids_all = np.concatenate([away_ids, home_ids])
    X_all = np.concatenate([Xnat, Xswap], axis=0)
    h_all = np.concatenate([h, a])
    a_all = np.concatenate([a, h])
    return (
        torch.from_numpy(home_ids_all.astype(np.int64)),
        torch.from_numpy(away_ids_all.astype(np.int64)),
        torch.from_numpy(X_all),
        torch.from_numpy(h_all.astype(np.int64)),
        torch.from_numpy(a_all.astype(np.int64)),
    )


def train_bivariate_model(
    train: pd.DataFrame, val_es: pd.DataFrame,
    team_idx: dict[str, int], config: BivariateConfig | None = None,
) -> tuple[BivariatePoissonModel, FeatureStats, dict]:
    cfg = config or BivariateConfig()
    _set_seed(cfg.seed)

    X_raw, _ = build_symmetric_rows(train, include_tier4=False)
    X_raw = np.nan_to_num(X_raw.astype(np.float64), nan=0.0)
    stats = FeatureStats(mean=X_raw.mean(axis=0), std=X_raw.std(axis=0))

    tr_tensors = _build_tensors_with_aug(train, team_idx, stats)
    va_tensors = _build_tensors(val_es, team_idx, stats)

    model = BivariatePoissonModel(
        n_teams=len(team_idx), embed_dim=cfg.embed_dim,
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    ds = TensorDataset(*tr_tensors)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)

    va_h, va_a, va_X, va_hg, va_ag = va_tensors

    best_val = float("inf")
    best_state: dict = {}
    patience_left = cfg.patience
    history: list[dict] = []

    for epoch in range(cfg.max_epochs):
        model.train()
        tloss = 0.0
        n_b = 0
        for b_h, b_a, b_X, b_hg, b_ag in loader:
            opt.zero_grad()
            log_lh, log_la, rho = model(b_h, b_a, b_X)
            log_prob = dc_log_prob_at_cell(log_lh, log_la, rho, b_hg, b_ag)
            loss = -log_prob.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            tloss += float(loss.item())
            n_b += 1
        sched.step()
        avg_tr = tloss / max(n_b, 1)

        model.eval()
        with torch.no_grad():
            log_lh, log_la, rho = model(va_h, va_a, va_X)
            log_prob = dc_log_prob_at_cell(log_lh, log_la, rho, va_hg, va_ag)
            val_loss = float(-log_prob.mean().item())
        history.append({"epoch": epoch, "train": avg_tr, "val_es": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("bivariate early_stopping", epoch=epoch, best_val=best_val)
                break
        if epoch % 5 == 0:
            log.info("bivariate epoch", epoch=epoch, train=avg_tr, val_es=val_loss)

    if best_state:
        model.load_state_dict(best_state)
    return model, stats, {"best_val_es": best_val, "history": history,
                          "n_epochs_run": len(history)}


def predict_lambda_rho(
    model: BivariatePoissonModel, matches: pd.DataFrame,
    team_idx: dict[str, int], stats: FeatureStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (lam_home, lam_away, rho) per match."""
    home_ids, away_ids, X, _, _ = _build_tensors(matches, team_idx, stats)
    model.eval()
    with torch.no_grad():
        log_lh, log_la, rho = model(home_ids, away_ids, X)
    return np.exp(log_lh.numpy()), np.exp(log_la.numpy()), rho.numpy()


def save_bivariate(
    model: BivariatePoissonModel, team_idx: dict[str, int],
    stats: FeatureStats, config: BivariateConfig, out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "team_idx.json").write_text(json.dumps(team_idx, indent=2))
    (out_dir / "feature_stats.json").write_text(json.dumps({
        "mean": stats.mean.tolist(), "std": stats.std.tolist(),
    }, indent=2))
    cfg = asdict(config)
    cfg["n_teams"] = len(team_idx)
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def load_bivariate(
    in_dir: Path,
) -> tuple[BivariatePoissonModel, dict[str, int], FeatureStats, BivariateConfig]:
    in_dir = Path(in_dir)
    raw = json.loads((in_dir / "config.json").read_text())
    n_teams = raw.pop("n_teams")
    raw["hidden_dims"] = tuple(raw["hidden_dims"])
    cfg = BivariateConfig(**raw)
    model = BivariatePoissonModel(
        n_teams=n_teams, embed_dim=cfg.embed_dim,
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )
    state = torch.load(in_dir / "weights.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    team_idx = json.loads((in_dir / "team_idx.json").read_text())
    sj = json.loads((in_dir / "feature_stats.json").read_text())
    return (
        model, team_idx,
        FeatureStats(mean=np.array(sj["mean"]), std=np.array(sj["std"])),
        cfg,
    )
