"""Monte Carlo tournament simulation from per-match joint goal matrices.

For each match: sample (home_goals, away_goals) from the DC-corrected joint.
Roll up over a tournament (groups + knockouts) and aggregate probabilities.

Used by ``scripts/predict_wc2026_groups.py`` and ``scripts/predict_wc2026_knockout.py``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from mondiali.model.dixon_coles import dixon_coles_correct, joint_matrix


def sample_match_scores(
    lam_h: float, lam_a: float, rho: float, *,
    n_sims: int, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample (home_goals, away_goals) n_sims times from DC-corrected joint."""
    m = joint_matrix(lam_h, lam_a)
    m = dixon_coles_correct(m, lam_h, lam_a, rho=rho)
    m = np.clip(m, 0.0, None)
    m = m / m.sum()
    flat = m.flatten()
    grid_size = m.shape[0]
    idx = rng.choice(flat.size, size=n_sims, p=flat)
    h = idx // grid_size
    a = idx % grid_size
    return h.astype(np.int32), a.astype(np.int32)


def points_for_result(home_goals: int, away_goals: int) -> tuple[int, int]:
    """Standard football points: 3 win / 1 draw / 0 loss. Returns (home_pts, away_pts)."""
    if home_goals > away_goals:
        return 3, 0
    if home_goals < away_goals:
        return 0, 3
    return 1, 1


def simulate_group(
    group_matches: list[dict], *,
    n_sims: int = 10000, seed: int = 42,
) -> pd.DataFrame:
    """Simulate a round-robin group n_sims times.

    Each entry in ``group_matches`` must have:
        team_a, team_b, lam_a, lam_b, rho

    Returns DataFrame indexed by team with columns:
        p_first, p_second, p_qualified (= p_first + p_second), p_eliminated,
        avg_points, avg_gd, avg_gf
    """
    rng = np.random.default_rng(seed)
    teams = sorted({m["team_a"] for m in group_matches} | {m["team_b"] for m in group_matches})
    n_teams = len(teams)
    idx_of = {t: i for i, t in enumerate(teams)}

    # Pre-sample scores for all matches
    match_samples: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    for m in group_matches:
        h, a = sample_match_scores(
            float(m["lam_a"]), float(m["lam_b"]), float(m["rho"]),
            n_sims=n_sims, rng=rng,
        )
        match_samples.append((idx_of[m["team_a"]], idx_of[m["team_b"]], h, a))

    points = np.zeros((n_sims, n_teams), dtype=np.int32)
    goals_for = np.zeros((n_sims, n_teams), dtype=np.int32)
    goals_against = np.zeros((n_sims, n_teams), dtype=np.int32)
    for ia, ib, h, a in match_samples:
        # Vectorized over sims
        a_wins = h > a
        b_wins = h < a
        draws = h == a
        points[a_wins, ia] += 3
        points[b_wins, ib] += 3
        points[draws, ia] += 1
        points[draws, ib] += 1
        goals_for[:, ia] += h
        goals_for[:, ib] += a
        goals_against[:, ia] += a
        goals_against[:, ib] += h

    # Ranking per sim: sort by (points desc, GD desc, GF desc, random tiebreak)
    gd = goals_for - goals_against
    # Tiebreak: stable sort key with random component for true ties
    tiebreak = rng.random((n_sims, n_teams))
    rank_keys = -(
        points.astype(np.float64) * 1e9
        + gd.astype(np.float64) * 1e5
        + goals_for.astype(np.float64) * 1e1
        + tiebreak
    )
    ranks = np.argsort(rank_keys, axis=1)
    # ranks[s, 0] = idx of 1st place in sim s, ranks[s, 1] = 2nd, ...
    first_counts = np.zeros(n_teams, dtype=np.int64)
    second_counts = np.zeros(n_teams, dtype=np.int64)
    for s in range(n_sims):
        first_counts[ranks[s, 0]] += 1
        second_counts[ranks[s, 1]] += 1

    df = pd.DataFrame({
        "team": teams,
        "p_first": first_counts / n_sims,
        "p_second": second_counts / n_sims,
        "avg_points": points.mean(axis=0),
        "avg_gd": gd.mean(axis=0),
        "avg_gf": goals_for.mean(axis=0),
    })
    df["p_qualified"] = df["p_first"] + df["p_second"]
    df["p_eliminated"] = 1.0 - df["p_qualified"]
    return df.sort_values("p_qualified", ascending=False).reset_index(drop=True)


def simulate_knockout_bracket(
    bracket: list[dict], match_predictor, *,
    n_sims: int = 10000, seed: int = 42,
) -> dict:
    """Simulate a knockout bracket given a callable ``match_predictor(home, away) -> (lam_h, lam_a, rho)``.

    ``bracket`` is a list-of-rounds: bracket[0] = list of (team_a, team_b) for round-of-16,
    bracket[1] is determined dynamically from round-of-16 winners, etc.

    Since the bracket depends on winners, this function builds a single-elimination
    structure where each round's pairings are positional: bracket[0] = [(t0, t1), (t2, t3), ...]
    and round k+1 pairs winners of (2i, 2i+1).

    Returns dict with per-team probabilities of reaching each round.
    """
    rng = np.random.default_rng(seed)
    round0 = bracket
    # Collect all 16 teams from round 0 pairings
    all_teams = []
    for pair in round0:
        all_teams.extend([pair["team_a"], pair["team_b"]])
    n_teams = len(all_teams)
    n_rounds = int(np.log2(n_teams))  # 16->4, 8->3, etc.

    # Reach probabilities per round (round 0 = R16 entry, ..., final round = winner)
    # Index: team -> count at each level
    reach_counts = {t: np.zeros(n_rounds + 1, dtype=np.int64) for t in all_teams}
    for t in all_teams:
        reach_counts[t][0] = n_sims  # all teams "reach" R16 by being in bracket

    for s in range(n_sims):
        survivors = [pair["team_a"] for pair in round0] + [pair["team_b"] for pair in round0]
        # Reorder so we can pair (0,1), (2,3), ...
        survivors = []
        for pair in round0:
            survivors.append(pair["team_a"])
            survivors.append(pair["team_b"])
        for r in range(1, n_rounds + 1):
            next_round = []
            for i in range(0, len(survivors), 2):
                t_a, t_b = survivors[i], survivors[i + 1]
                lam_a, lam_b, rho = match_predictor(t_a, t_b)
                h, a = sample_match_scores(lam_a, lam_b, rho, n_sims=1, rng=rng)
                if h[0] > a[0]:
                    winner = t_a
                elif h[0] < a[0]:
                    winner = t_b
                else:
                    # Penalty shootout — 50/50
                    winner = t_a if rng.random() < 0.5 else t_b
                next_round.append(winner)
                reach_counts[winner][r] += 1
            survivors = next_round

    results = []
    for t in all_teams:
        rc = reach_counts[t] / n_sims
        results.append({
            "team": t,
            "p_round_of_16": float(rc[0]),
            **{f"p_round_{r}": float(rc[r]) for r in range(1, n_rounds + 1)},
        })
    out = pd.DataFrame(results)
    # Sort by deepest stage
    final_col = f"p_round_{n_rounds}"
    out = out.sort_values(final_col, ascending=False).reset_index(drop=True)
    return {"per_team": out, "n_rounds": n_rounds, "n_sims": n_sims}
