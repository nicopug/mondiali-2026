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


def simulate_tournament(
    groups_cfg: dict[str, list[str]],
    pair_lambdas: dict[tuple[str, str], tuple[float, float]],
    rho: float,
    *,
    elo_dict: dict[str, float],
    n_sims: int = 10000,
    seed: int = 42,
) -> pd.DataFrame:
    """End-to-end Monte Carlo: groups + knockout in one simulation pass.

    Returns per-team probabilities for each tournament stage. Unlike the
    two-step approach where knockout is conditional on a fixed 32-team list,
    this propagates group uncertainty into knockout probabilities (so a team
    that's borderline qualifier gets P(reach R16) = P(qualif) * P(advance | qualif)).

    Bracket pairing strategy: 32 qualifiers (12 winners + 12 runners-up + 8 best
    thirds) seeded by Elo descending; seed 1 vs 32, 2 vs 31, etc. Pure positional
    pairing — does NOT enforce FIFA's "same-group teams can't meet in R32" rule
    (would require encoding the 495-scenario chart).
    """
    rng = np.random.default_rng(seed)
    all_teams = [t for group in groups_cfg.values() for t in group]
    team_to_idx = {t: i for i, t in enumerate(all_teams)}
    n_teams = len(all_teams)

    # Per-team counters
    stage_counts = {
        "p_first": np.zeros(n_teams, dtype=np.int64),
        "p_second": np.zeros(n_teams, dtype=np.int64),
        "p_third": np.zeros(n_teams, dtype=np.int64),
        "p_third_advanced": np.zeros(n_teams, dtype=np.int64),
        "p_qualified_r32": np.zeros(n_teams, dtype=np.int64),
        "p_round_of_16": np.zeros(n_teams, dtype=np.int64),
        "p_quarterfinal": np.zeros(n_teams, dtype=np.int64),
        "p_semifinal": np.zeros(n_teams, dtype=np.int64),
        "p_final": np.zeros(n_teams, dtype=np.int64),
        "p_winner": np.zeros(n_teams, dtype=np.int64),
    }
    points_sum = np.zeros(n_teams, dtype=np.float64)
    gd_sum = np.zeros(n_teams, dtype=np.float64)

    # Pre-sample group match scores for vectorization
    from collections import defaultdict
    group_match_scores: dict[str, list[tuple[str, str, np.ndarray, np.ndarray]]] = defaultdict(list)
    for group_letter, teams in groups_cfg.items():
        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                t_a, t_b = teams[i], teams[j]
                lam_a, lam_b = pair_lambdas[(t_a, t_b)]
                h, a = sample_match_scores(lam_a, lam_b, rho, n_sims=n_sims, rng=rng)
                group_match_scores[group_letter].append((t_a, t_b, h, a))

    # For each sim, compute group standings + qualifications + knockout
    for s in range(n_sims):
        # Group standings
        group_standings: dict[str, list[tuple[str, int, int, int]]] = {}
        thirds_pool: list[tuple[str, int, int, int]] = []  # (team, pts, gd, gf)
        qualifiers_winners: list[str] = []
        qualifiers_runnersup: list[str] = []

        for group_letter, teams in groups_cfg.items():
            pts = {t: 0 for t in teams}
            gf = {t: 0 for t in teams}
            ga = {t: 0 for t in teams}
            for t_a, t_b, h, a in group_match_scores[group_letter]:
                ha, ab = int(h[s]), int(a[s])
                if ha > ab:
                    pts[t_a] += 3
                elif ha < ab:
                    pts[t_b] += 3
                else:
                    pts[t_a] += 1; pts[t_b] += 1
                gf[t_a] += ha; ga[t_a] += ab
                gf[t_b] += ab; ga[t_b] += ha
            # Tiebreak: pts desc, gd desc, gf desc, random
            tiebreak = {t: rng.random() for t in teams}
            ranked = sorted(
                teams,
                key=lambda t: (-pts[t], -(gf[t] - ga[t]), -gf[t], tiebreak[t]),
            )
            group_standings[group_letter] = [
                (t, pts[t], gf[t] - ga[t], gf[t]) for t in ranked
            ]
            qualifiers_winners.append(ranked[0])
            qualifiers_runnersup.append(ranked[1])
            thirds_pool.append((ranked[2], pts[ranked[2]],
                                gf[ranked[2]] - ga[ranked[2]], gf[ranked[2]]))
            # Update counters + aggregates
            for rank_idx, t in enumerate(ranked):
                points_sum[team_to_idx[t]] += pts[t]
                gd_sum[team_to_idx[t]] += gf[t] - ga[t]
                if rank_idx == 0:
                    stage_counts["p_first"][team_to_idx[t]] += 1
                elif rank_idx == 1:
                    stage_counts["p_second"][team_to_idx[t]] += 1
                elif rank_idx == 2:
                    stage_counts["p_third"][team_to_idx[t]] += 1

        # Pick 8 best thirds
        thirds_pool.sort(key=lambda x: (-x[1], -x[2], -x[3], rng.random()))
        best_thirds = [x[0] for x in thirds_pool[:8]]
        for t in best_thirds:
            stage_counts["p_third_advanced"][team_to_idx[t]] += 1

        qualifiers = qualifiers_winners + qualifiers_runnersup + best_thirds
        for t in qualifiers:
            stage_counts["p_qualified_r32"][team_to_idx[t]] += 1

        # Seed by Elo descending → bracket 1 vs 32
        seeded = sorted(qualifiers, key=lambda t: -elo_dict.get(t, 1500.0))
        # R32 → R16 → QF → SF → Final
        survivors = seeded
        round_order = ["p_round_of_16", "p_quarterfinal",
                       "p_semifinal", "p_final", "p_winner"]
        for round_name in round_order:
            next_round = []
            n_pairs = len(survivors) // 2
            for i in range(n_pairs):
                t_a, t_b = survivors[i], survivors[len(survivors) - 1 - i]
                lam_a, lam_b = pair_lambdas[(t_a, t_b)]
                h, a = sample_match_scores(lam_a, lam_b, rho, n_sims=1, rng=rng)
                if h[0] > a[0]:
                    winner = t_a
                elif h[0] < a[0]:
                    winner = t_b
                else:
                    winner = t_a if rng.random() < 0.5 else t_b
                next_round.append(winner)
            for t in next_round:
                stage_counts[round_name][team_to_idx[t]] += 1
            # Re-sort survivors by Elo to keep a deterministic bracket each round
            survivors = sorted(next_round, key=lambda t: -elo_dict.get(t, 1500.0))

    # Build output DataFrame
    rows = []
    for i, t in enumerate(all_teams):
        # Find team's group
        team_group = next(g for g, ts in groups_cfg.items() if t in ts)
        row = {
            "team": t, "group": team_group,
            "avg_points": float(points_sum[i] / n_sims),
            "avg_gd": float(gd_sum[i] / n_sims),
        }
        for k, c in stage_counts.items():
            row[k] = float(c[i] / n_sims)
        rows.append(row)
    df = pd.DataFrame(rows)
    return df.sort_values(["p_winner", "p_qualified_r32"], ascending=False).reset_index(drop=True)


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
