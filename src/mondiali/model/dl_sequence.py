"""Tier 7-L2: team-embedding + sequence GRU over team history.

Extends L1 by replacing the hand-crafted form-5 features with a learned
representation from a GRU over the team's last HIST_LEN matches (strictly
pre-target-date, anti-leakage).

Sequence element per past match (6 dim):
    [score_for, score_against, opponent_elo, is_home, log_days_ago, comp_imp]
all z-score normalized (stats computed on training rows).
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
HIST_LEN = 8
HIST_FEAT_NAMES = (
    "score_for", "score_against", "opp_elo",
    "is_home", "log_days_ago", "comp_imp",
)
N_HIST_FEATS = len(HIST_FEAT_NAMES)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@dataclass
class SeqConfig:
    embed_dim: int = 16
    seq_hidden: int = 32
    hidden_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch_size: int = 256  # smaller than L1: sequences are heavier
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


def _team_long(matches: pd.DataFrame) -> pd.DataFrame:
    """Long-form per-team rows ordered by team+date for fast history lookup."""
    home = pd.DataFrame({
        "team": matches["home_team"].to_numpy(),
        "date": pd.to_datetime(matches["date"]).to_numpy(),
        "score_for": matches["home_score"].astype(float).to_numpy(),
        "score_against": matches["away_score"].astype(float).to_numpy(),
        "opp_elo": matches["away_elo_before"].astype(float).to_numpy(),
        "is_home": (~matches["neutral"].astype(bool)).astype(float).to_numpy(),
        "comp_imp": matches["competition_importance"].astype(float).to_numpy(),
    })
    away = pd.DataFrame({
        "team": matches["away_team"].to_numpy(),
        "date": pd.to_datetime(matches["date"]).to_numpy(),
        "score_for": matches["away_score"].astype(float).to_numpy(),
        "score_against": matches["home_score"].astype(float).to_numpy(),
        "opp_elo": matches["home_elo_before"].astype(float).to_numpy(),
        "is_home": 0.0,
        "comp_imp": matches["competition_importance"].astype(float).to_numpy(),
    })
    long = pd.concat([home, away], ignore_index=True)
    return long.sort_values(["team", "date"]).reset_index(drop=True)


def build_history_tensor(
    target_matches: pd.DataFrame,
    history_universe: pd.DataFrame,
    hist_len: int = HIST_LEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Build (n_targets, 2, hist_len, 6) feature tensor + mask (n_targets, 2, hist_len).

    Axis 1: 0=home perspective, 1=away perspective.
    history_universe is the full corpus of matches we can look back into.
    """
    long = _team_long(history_universe)
    by_team: dict[str, pd.DataFrame] = {
        team: grp.reset_index(drop=True) for team, grp in long.groupby("team")
    }

    n = len(target_matches)
    hist = np.zeros((n, 2, hist_len, N_HIST_FEATS), dtype=np.float32)
    mask = np.zeros((n, 2, hist_len), dtype=np.float32)

    target_dates = pd.to_datetime(target_matches["date"]).to_numpy()
    target_home = target_matches["home_team"].to_numpy()
    target_away = target_matches["away_team"].to_numpy()

    for i in range(n):
        d = target_dates[i]
        for side, team in [(0, target_home[i]), (1, target_away[i])]:
            grp = by_team.get(team)
            if grp is None or len(grp) == 0:
                continue
            # strict pre-date
            past = grp[grp["date"].to_numpy() < d]
            if len(past) == 0:
                continue
            past = past.iloc[-hist_len:]  # most recent hist_len
            k = len(past)
            log_days = np.log(((d - past["date"].to_numpy()).astype("timedelta64[D]")
                               .astype(np.float64)) + 1.0)
            row = np.stack([
                past["score_for"].to_numpy(np.float32),
                past["score_against"].to_numpy(np.float32),
                past["opp_elo"].to_numpy(np.float32),
                past["is_home"].to_numpy(np.float32),
                log_days.astype(np.float32),
                past["comp_imp"].to_numpy(np.float32),
            ], axis=-1)
            # Place at the END of the time axis (right-aligned, easier for GRU)
            hist[i, side, hist_len - k:hist_len] = row
            mask[i, side, hist_len - k:hist_len] = 1.0
    return hist, mask


def _build_tab_features(
    matches: pd.DataFrame, stats: FeatureStats | None = None,
) -> np.ndarray:
    X, _ = build_symmetric_rows(matches, include_tier4=False)
    X = np.nan_to_num(X.astype(np.float64), nan=0.0)
    if stats is not None:
        X = stats.transform(X)
    # build_symmetric_rows produces 2 rows/match; we keep BOTH for symmetric training
    return X.astype(np.float32)


def _hist_stats_from_train(hist_tensor: np.ndarray, mask: np.ndarray) -> FeatureStats:
    flat = hist_tensor.reshape(-1, N_HIST_FEATS)
    m = mask.reshape(-1).astype(bool)
    valid = flat[m]
    return FeatureStats(
        mean=valid.mean(axis=0).astype(np.float32),
        std=valid.std(axis=0).astype(np.float32),
    )


def _apply_hist_stats(hist: np.ndarray, mask: np.ndarray, s: FeatureStats) -> np.ndarray:
    out = (hist - s.mean) / np.where(s.std > 1e-8, s.std, 1.0)
    # zero-out masked positions
    out = out * mask[..., None]
    return out.astype(np.float32)


class SequencePoissonModel(nn.Module):
    def __init__(
        self,
        n_teams: int,
        n_features: int = len(SYMMETRIC_FEATURES),
        embed_dim: int = 16,
        seq_hidden: int = 32,
        hidden_dims: tuple[int, ...] = (128, 64),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.team_emb = nn.Embedding(n_teams + 1, embed_dim, padding_idx=UNK_IDX)
        nn.init.normal_(self.team_emb.weight, mean=0.0, std=0.1)
        with torch.no_grad():
            self.team_emb.weight[UNK_IDX].zero_()
        self.seq_encoder = nn.GRU(
            input_size=N_HIST_FEATS,
            hidden_size=seq_hidden,
            batch_first=True,
        )
        self.embed_dim = embed_dim
        self.seq_hidden = seq_hidden
        self.n_features = n_features
        layers: list[nn.Module] = []
        prev = 2 * embed_dim + 2 * seq_hidden + n_features
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)])
            prev = h
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 1)

    def _encode_seq(self, seq: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # seq: (batch, hist_len, n_feats). lengths: (batch,) actual lengths.
        # Use packed sequence for correctness when masks vary.
        # For simplicity here: just run GRU on full padded seq and take last hidden.
        # Zero rows in seq mean "no event" — GRU handles them as benign 0 input.
        _, h_last = self.seq_encoder(seq)
        return h_last.squeeze(0)

    def forward(
        self,
        team_ids: torch.Tensor, opp_ids: torch.Tensor,
        team_hist: torch.Tensor, opp_hist: torch.Tensor,
        team_lens: torch.Tensor, opp_lens: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        et = self.team_emb(team_ids)
        eo = self.team_emb(opp_ids)
        h_team = self._encode_seq(team_hist, team_lens)
        h_opp = self._encode_seq(opp_hist, opp_lens)
        x = torch.cat([et, eo, h_team, h_opp, features], dim=-1)
        h = self.mlp(x)
        return self.head(h).squeeze(-1)


def _poisson_nll(log_lambda: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.poisson_nll_loss(log_lambda, y, log_input=True, full=False, reduction="mean")


def _build_dataset(
    matches: pd.DataFrame,
    history_universe: pd.DataFrame,
    team_idx: dict[str, int],
    tab_stats: FeatureStats | None,
    hist_stats: FeatureStats | None,
) -> tuple[torch.Tensor, ...]:
    """Build all tensors needed by SequencePoissonModel for symmetric training.

    Returns (team_ids, opp_ids, team_hist, opp_hist, team_lens, opp_lens, features, y)
    where each tensor has length 2 * len(matches).
    """
    n = len(matches)
    # Tab features: 2 rows/match already
    X = _build_tab_features(matches, stats=tab_stats)
    _, y = build_symmetric_rows(matches, include_tier4=False)

    # History: (n, 2 sides, hist_len, n_feats)
    hist, mask = build_history_tensor(matches, history_universe)
    if hist_stats is not None:
        hist = _apply_hist_stats(hist, mask, hist_stats)

    home_ids = matches["home_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()
    away_ids = matches["away_team"].map(team_idx).fillna(UNK_IDX).astype(int).to_numpy()

    # Symmetric: even rows = home-perspective, odd = away-perspective
    team_ids = np.empty(2 * n, dtype=np.int64)
    opp_ids = np.empty(2 * n, dtype=np.int64)
    team_ids[0::2] = home_ids
    team_ids[1::2] = away_ids
    opp_ids[0::2] = away_ids
    opp_ids[1::2] = home_ids

    team_hist = np.empty((2 * n, HIST_LEN, N_HIST_FEATS), dtype=np.float32)
    opp_hist = np.empty((2 * n, HIST_LEN, N_HIST_FEATS), dtype=np.float32)
    team_hist[0::2] = hist[:, 0]
    team_hist[1::2] = hist[:, 1]
    opp_hist[0::2] = hist[:, 1]
    opp_hist[1::2] = hist[:, 0]

    team_lens = np.empty(2 * n, dtype=np.int64)
    opp_lens = np.empty(2 * n, dtype=np.int64)
    team_lens[0::2] = mask[:, 0].sum(axis=-1).astype(np.int64)
    team_lens[1::2] = mask[:, 1].sum(axis=-1).astype(np.int64)
    opp_lens[0::2] = mask[:, 1].sum(axis=-1).astype(np.int64)
    opp_lens[1::2] = mask[:, 0].sum(axis=-1).astype(np.int64)

    return (
        torch.from_numpy(team_ids),
        torch.from_numpy(opp_ids),
        torch.from_numpy(team_hist),
        torch.from_numpy(opp_hist),
        torch.from_numpy(team_lens),
        torch.from_numpy(opp_lens),
        torch.from_numpy(X),
        torch.from_numpy(y.astype(np.float32)),
    )


def train_seq_model(
    train: pd.DataFrame,
    val_es: pd.DataFrame,
    history_universe: pd.DataFrame,
    team_idx: dict[str, int],
    config: SeqConfig | None = None,
) -> tuple[SequencePoissonModel, FeatureStats, FeatureStats, dict]:
    cfg = config or SeqConfig()
    _set_seed(cfg.seed)

    X_tr, _ = build_symmetric_rows(train, include_tier4=False)
    X_tr = np.nan_to_num(X_tr.astype(np.float64), nan=0.0)
    tab_stats = FeatureStats(mean=X_tr.mean(axis=0), std=X_tr.std(axis=0))
    hist_tr, mask_tr = build_history_tensor(train, history_universe)
    hist_stats = _hist_stats_from_train(hist_tr, mask_tr)
    log.info("seq stats computed", tab_dim=tab_stats.mean.shape[0],
             hist_dim=hist_stats.mean.shape[0])

    tr_tensors = _build_dataset(train, history_universe, team_idx, tab_stats, hist_stats)
    va_tensors = _build_dataset(val_es, history_universe, team_idx, tab_stats, hist_stats)
    log.info("seq data built", n_train=len(tr_tensors[-1]), n_val=len(va_tensors[-1]))

    model = SequencePoissonModel(
        n_teams=len(team_idx),
        embed_dim=cfg.embed_dim,
        seq_hidden=cfg.seq_hidden,
        hidden_dims=cfg.hidden_dims,
        dropout=cfg.dropout,
    )
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.max_epochs)

    ds = TensorDataset(*tr_tensors)
    gen = torch.Generator().manual_seed(cfg.seed)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, generator=gen)

    best_val = float("inf")
    best_state: dict = {}
    patience_left = cfg.patience
    history: list[dict] = []

    va_team, va_opp, va_th, va_oh, va_tl, va_ol, va_X, va_y = va_tensors

    for epoch in range(cfg.max_epochs):
        model.train()
        train_loss = 0.0
        n_b = 0
        for batch in loader:
            b_team, b_opp, b_th, b_oh, b_tl, b_ol, b_X, b_y = batch
            opt.zero_grad()
            pred = model(b_team, b_opp, b_th, b_oh, b_tl, b_ol, b_X)
            loss = _poisson_nll(pred, b_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            train_loss += float(loss.item())
            n_b += 1
        sched.step()
        avg_tr = train_loss / max(n_b, 1)

        model.eval()
        with torch.no_grad():
            pred = model(va_team, va_opp, va_th, va_oh, va_tl, va_ol, va_X)
            val_loss = float(_poisson_nll(pred, va_y).item())
        history.append({"epoch": epoch, "train": avg_tr, "val_es": val_loss})

        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            patience_left = cfg.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                log.info("seq early_stopping", epoch=epoch, best_val=best_val)
                break
        if epoch % 5 == 0:
            log.info("seq epoch", epoch=epoch, train=avg_tr, val_es=val_loss)

    if best_state:
        model.load_state_dict(best_state)
    return model, tab_stats, hist_stats, {
        "best_val_es": best_val, "history": history, "n_epochs_run": len(history),
    }


def predict_lambda(
    model: SequencePoissonModel,
    matches: pd.DataFrame,
    history_universe: pd.DataFrame,
    team_idx: dict[str, int],
    tab_stats: FeatureStats,
    hist_stats: FeatureStats,
) -> tuple[np.ndarray, np.ndarray]:
    tensors = _build_dataset(matches, history_universe, team_idx, tab_stats, hist_stats)
    team_ids, opp_ids, team_hist, opp_hist, team_lens, opp_lens, X, _ = tensors
    model.eval()
    with torch.no_grad():
        log_lam = model(team_ids, opp_ids, team_hist, opp_hist, team_lens, opp_lens, X).numpy()
    lam = np.exp(log_lam)
    return lam[0::2], lam[1::2]


def save_seq_model(
    model: SequencePoissonModel,
    team_idx: dict[str, int],
    tab_stats: FeatureStats, hist_stats: FeatureStats,
    config: SeqConfig, out_dir: Path,
) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out_dir / "weights.pt")
    (out_dir / "team_idx.json").write_text(json.dumps(team_idx, indent=2))
    (out_dir / "tab_stats.json").write_text(json.dumps({
        "mean": tab_stats.mean.tolist(), "std": tab_stats.std.tolist(),
    }, indent=2))
    (out_dir / "hist_stats.json").write_text(json.dumps({
        "mean": hist_stats.mean.tolist(), "std": hist_stats.std.tolist(),
    }, indent=2))
    cfg = asdict(config)
    cfg["n_teams"] = len(team_idx)
    cfg["hist_len"] = HIST_LEN
    cfg["n_hist_feats"] = N_HIST_FEATS
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))


def load_seq_model(
    in_dir: Path,
) -> tuple[SequencePoissonModel, dict[str, int], FeatureStats, FeatureStats, SeqConfig]:
    in_dir = Path(in_dir)
    raw = json.loads((in_dir / "config.json").read_text())
    n_teams = raw.pop("n_teams")
    raw.pop("hist_len", None)
    raw.pop("n_hist_feats", None)
    raw["hidden_dims"] = tuple(raw["hidden_dims"])
    cfg = SeqConfig(**raw)
    model = SequencePoissonModel(
        n_teams=n_teams, embed_dim=cfg.embed_dim, seq_hidden=cfg.seq_hidden,
        hidden_dims=cfg.hidden_dims, dropout=cfg.dropout,
    )
    state = torch.load(in_dir / "weights.pt", weights_only=True, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    team_idx = json.loads((in_dir / "team_idx.json").read_text())
    tj = json.loads((in_dir / "tab_stats.json").read_text())
    hj = json.loads((in_dir / "hist_stats.json").read_text())
    return (
        model, team_idx,
        FeatureStats(mean=np.array(tj["mean"]), std=np.array(tj["std"])),
        FeatureStats(mean=np.array(hj["mean"]), std=np.array(hj["std"])),
        cfg,
    )
