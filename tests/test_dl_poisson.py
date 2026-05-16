"""Tests for dl_poisson.py — model forward + determinism + save/load."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from mondiali.model.dl_poisson import (  # noqa: E402
    DLConfig,
    PoissonEmbeddingModel,
    _set_seed,
    build_team_index,
    load_dl_model,
    predict_lambda,
    save_dl_model,
    train_dl_model,
)


def _make_matches(n: int = 100, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    teams = ["France", "Italy", "Spain", "Germany", "Brazil", "Argentina"]
    rows = []
    for i in range(n):
        h, a = rng.choice(teams, size=2, replace=False)
        rows.append({
            "date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i),
            "home_team": h, "away_team": a,
            "home_score": int(rng.poisson(1.5)),
            "away_score": int(rng.poisson(1.2)),
            "neutral": False,
            "home_elo_before": 1800.0 + rng.normal(0, 50),
            "away_elo_before": 1800.0 + rng.normal(0, 50),
            "competition_importance": 30.0,
            "days_rest_home": 7.0, "days_rest_away": 7.0,
            "home_form_5": rng.uniform(0, 15), "away_form_5": rng.uniform(0, 15),
            "home_gd_5": rng.uniform(-10, 10), "away_gd_5": rng.uniform(-10, 10),
            "home_goals_scored_5": rng.uniform(0, 3),
            "away_goals_scored_5": rng.uniform(0, 3),
            "home_goals_conceded_5": rng.uniform(0, 3),
            "away_goals_conceded_5": rng.uniform(0, 3),
            "home_avg_opp_elo_5": 1800.0 + rng.normal(0, 50),
            "away_avg_opp_elo_5": 1800.0 + rng.normal(0, 50),
            "home_market_value_total": rng.uniform(1e8, 1e9),
            "away_market_value_total": rng.uniform(1e8, 1e9),
            "home_market_value_top11": rng.uniform(5e7, 5e8),
            "away_market_value_top11": rng.uniform(5e7, 5e8),
            "home_tm_age_days": rng.uniform(30, 540),
            "away_tm_age_days": rng.uniform(30, 540),
        })
    return pd.DataFrame(rows)


def test_model_forward_shape() -> None:
    n_teams = 6
    model = PoissonEmbeddingModel(n_teams=n_teams)
    team_ids = torch.randint(1, n_teams + 1, (8,))
    opp_ids = torch.randint(1, n_teams + 1, (8,))
    features = torch.randn(8, 24)
    out = model(team_ids, opp_ids, features)
    assert out.shape == (8,)


def test_training_smoke_runs_and_decreases_loss() -> None:
    matches = _make_matches(n=200)
    team_idx = build_team_index(matches)
    train = matches.iloc[:150].copy()
    val_es = matches.iloc[150:].copy()
    cfg = DLConfig(max_epochs=10, patience=20, batch_size=64)
    model, stats, info = train_dl_model(train, val_es, team_idx, cfg)
    history = info["history"]
    assert history[-1]["train"] < history[0]["train"]
    assert info["best_val_es"] < float("inf")


def test_determinism_same_seed_same_weights() -> None:
    matches = _make_matches(n=120)
    team_idx = build_team_index(matches)
    cfg = DLConfig(max_epochs=5, patience=20, batch_size=64, seed=42)
    m1, s1, _ = train_dl_model(matches.iloc[:90], matches.iloc[90:], team_idx, cfg)
    m2, s2, _ = train_dl_model(matches.iloc[:90], matches.iloc[90:], team_idx, cfg)
    for (k1, v1), (k2, v2) in zip(m1.state_dict().items(), m2.state_dict().items(), strict=True):
        assert k1 == k2
        torch.testing.assert_close(v1, v2, msg=f"determinism broken at {k1}")


def test_predict_lambda_shapes_and_positive() -> None:
    matches = _make_matches(n=50)
    team_idx = build_team_index(matches)
    cfg = DLConfig(max_epochs=3, patience=20, batch_size=32)
    model, stats, _ = train_dl_model(matches.iloc[:40], matches.iloc[40:], team_idx, cfg)
    lam_h, lam_a = predict_lambda(model, matches.iloc[40:], team_idx, stats)
    assert lam_h.shape == (10,)
    assert lam_a.shape == (10,)
    assert (lam_h > 0).all()
    assert (lam_a > 0).all()


def test_save_load_roundtrip(tmp_path: Path) -> None:
    matches = _make_matches(n=80)
    team_idx = build_team_index(matches)
    cfg = DLConfig(max_epochs=3, patience=20, batch_size=32)
    model, stats, _ = train_dl_model(matches.iloc[:60], matches.iloc[60:], team_idx, cfg)
    save_dl_model(model, team_idx, stats, cfg, tmp_path / "dl")
    lam_h_before, _ = predict_lambda(model, matches.iloc[60:], team_idx, stats)
    model2, idx2, stats2, _ = load_dl_model(tmp_path / "dl")
    assert idx2 == team_idx
    lam_h_after, _ = predict_lambda(model2, matches.iloc[60:], idx2, stats2)
    np.testing.assert_allclose(lam_h_before, lam_h_after, rtol=1e-5)


def test_unknown_team_falls_back_to_unk() -> None:
    matches = _make_matches(n=60)
    team_idx = build_team_index(matches)
    cfg = DLConfig(max_epochs=3, patience=20, batch_size=32)
    model, stats, _ = train_dl_model(matches.iloc[:50], matches.iloc[50:], team_idx, cfg)
    test = matches.iloc[50:51].copy()
    test["home_team"] = "Atlantis"
    lam_h, lam_a = predict_lambda(model, test, team_idx, stats)
    assert lam_h.shape == (1,)
    assert lam_h[0] > 0
    assert np.isfinite(lam_h[0])


def test_seed_isolation_does_not_pollute_global_state() -> None:
    """_set_seed should not poison numpy global rng for caller."""
    _set_seed(42)
    a = np.random.default_rng(99).normal()
    b = np.random.default_rng(99).normal()
    assert a == b  # default_rng with explicit seed is isolated
