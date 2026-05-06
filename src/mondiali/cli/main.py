"""Entry point Typer CLI per il package `mondiali`.

Comandi disponibili in STEP 1:
    mondiali ingest        Download + parsing + Elo history -> matches.parquet
    mondiali baseline      Fit PriorBaseline su training set, report log-loss
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import structlog
import typer

from mondiali.config import CONFIG
from mondiali.data.ingestion import build_processed_matches, download_international_results
from mondiali.data.scope import compute_tier3_scope
from mondiali.data.transfermarkt import scrape_all
from mondiali.model.elo_logistic import EloLogisticBaseline
from mondiali.training.baseline_prior import PriorBaseline
from mondiali.training.evaluate import log_loss_1x2
from mondiali.training.train import train_tier1_pipeline, train_tier2_pipeline

app = typer.Typer(add_completion=False, no_args_is_help=True)
log = structlog.get_logger(__name__)


@app.command()
def ingest(
    force: bool = typer.Option(False, "--force", help="Re-download anche se presente"),
) -> None:
    """Scarica international_results e produce `matches.parquet`."""
    raw_csv = CONFIG.data_raw / "results.csv"
    download_international_results(raw_csv, force=force)
    processed_path = CONFIG.data_processed / "matches.parquet"
    build_processed_matches(raw_csv, processed_path)
    typer.echo(f"OK - processed matches written to {processed_path}")


@app.command()
def baseline(
    train_start: str = typer.Option("2002-01-01", help="Inizio training set"),
    train_end: str = typer.Option("2018-12-31", help="Fine training (incluso)"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30", help="Fine validation (pre-WC2022)"),
) -> None:
    """Fit PriorBaseline su training, valuta su validation. Tier 0 floor."""
    processed = CONFIG.data_processed / "matches.parquet"
    if not processed.exists():
        typer.echo("matches.parquet non trovato - esegui `mondiali ingest` prima", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(processed)
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]

    typer.echo(f"Train: {len(train)} matches ({train_start} -> {train_end})")
    typer.echo(f"Val:   {len(val)} matches ({val_start} -> {val_end})")

    model = PriorBaseline()
    model.fit(train)
    assert model.prior_ is not None
    typer.echo(f"Prior 1/X/2 (dal training): {np.round(model.prior_, 4).tolist()}")

    val_probs = model.predict_proba(val)
    val_loss = log_loss_1x2(val, val_probs)
    typer.echo(f"Validation log-loss (Tier 0 prior baseline): {val_loss:.4f}")


@app.command(name="train-elo")
def train_elo(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2018-12-31"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30"),
) -> None:
    """Fit Elo-only logistic baseline, report log-loss su validation."""
    processed = CONFIG.data_processed / "matches.parquet"
    if not processed.exists():
        typer.echo("matches.parquet non trovato - esegui `mondiali ingest` prima", err=True)
        raise typer.Exit(1)

    df = pd.read_parquet(processed)
    df["date"] = pd.to_datetime(df["date"])
    train = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
    val = df[(df["date"] >= val_start) & (df["date"] <= val_end)]
    if train.empty or val.empty:
        typer.echo("Train o val vuoti - controlla i range date", err=True)
        raise typer.Exit(1)

    typer.echo(f"Train: {len(train)} matches ({train_start} -> {train_end})")
    typer.echo(f"Val:   {len(val)} matches ({val_start} -> {val_end})")

    model = EloLogisticBaseline().fit(train)
    val_probs = model.predict_proba(val)
    val_loss = log_loss_1x2(val, val_probs)
    typer.echo(f"Elo-only logistic validation log-loss: {val_loss:.4f}")
    assert model.model_ is not None
    coef = model.model_.coef_[0]
    typer.echo(f"Coefficienti: elo_diff={coef[0]:.6f}, is_neutral={coef[1]:.5f}")


@app.command(name="train-tier1")
def train_tier1(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2018-12-31"),
    val_start: str = typer.Option("2019-01-01"),
    val_end: str = typer.Option("2022-06-30"),
    save_model: bool = typer.Option(False, "--save", help="Salva il modello in models/tier1/"),
) -> None:
    """Addestra Tier 1 (XGBoost Poisson + Dixon-Coles), report log-loss."""
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_tier1_pipeline(
        parquet_path=parquet,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
    )
    typer.echo(f"Train: {result['n_train']} | Val: {result['n_val']}")
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(
        f"lambda_home_mean: {result['lambda_home_mean']:.3f} | "
        f"lambda_away_mean: {result['lambda_away_mean']:.3f}"
    )
    typer.echo(
        f"Tier 1 validation log-loss (1X2 calibrated by DC only): "
        f"{result['val_log_loss_1x2']:.4f}"
    )

    if save_model:
        out = CONFIG.models_dir / "tier1" / "xgb_poisson.json"
        result["model"].save(out)
        typer.echo(f"Model saved: {out}")


@app.command(name="train-tier2")
def train_tier2(
    train_start: str = typer.Option("2002-01-01"),
    train_end: str = typer.Option("2016-12-31"),
    val_es_start: str = typer.Option("2017-01-01"),
    val_es_end: str = typer.Option("2017-12-31"),
    val_calib_start: str = typer.Option("2018-01-01"),
    val_calib_end: str = typer.Option("2018-12-31"),
    val_gate_start: str = typer.Option("2019-01-01"),
    val_gate_end: str = typer.Option("2022-06-30"),
    save_model: str = typer.Option("", "--save-model", help="Path JSON dove salvare il modello"),
    save_calibrator: str = typer.Option(
        "", "--save-calibrator", help="Path JSON dove salvare il calibrator"
    ),
) -> None:
    """Addestra Tier 2 (XGBoost Poisson + DC + isotonic calibration)."""
    parquet = CONFIG.data_processed / "matches.parquet"
    result = train_tier2_pipeline(
        parquet_path=parquet,
        train_start=train_start, train_end=train_end,
        val_es_start=val_es_start, val_es_end=val_es_end,
        val_calib_start=val_calib_start, val_calib_end=val_calib_end,
        val_gate_start=val_gate_start, val_gate_end=val_gate_end,
    )
    typer.echo(
        f"Splits: train={result['n_train']} val_es={result['n_val_es']} "
        f"val_calib={result['n_val_calib']} val_gate={result['n_val_gate']}"
    )
    typer.echo(f"Dixon-Coles rho: {result['rho']:.4f}")
    typer.echo(f"Tier 2 RAW   log-loss: {result['val_log_loss_raw']:.4f}")
    typer.echo(f"Tier 2 CALIB log-loss: {result['val_log_loss_calib']:.4f}")
    typer.echo(f"Brier before: {result['brier_before']:.4f}")
    typer.echo(f"Brier after:  {result['brier_after']:.4f}")

    if save_model:
        result["model"].save(Path(save_model))
        typer.echo(f"Model saved: {save_model}")
    if save_calibrator:
        result["calibrator"].save(Path(save_calibrator))
        typer.echo(f"Calibrator saved: {save_calibrator}")


@app.command(name="tm-scrape")
def tm_scrape(
    start_year: int = typer.Option(2014, help="Anno iniziale snapshot"),
    end_year: int = typer.Option(2025, help="Anno finale snapshot incluso"),
    scope_file: str = typer.Option(
        "", "--scope-file",
        help="Path JSON con lista nazioni; se vuoto, computa da matches.parquet",
    ),
) -> None:
    """Scrape Transfermarkt market values via Wayback Machine per Tier 3.

    Quando ``--scope-file`` è omesso, computa lo scope da ``matches.parquet`` e lo
    salva in ``data/processed/tier3_scope.json`` come effetto collaterale.
    """
    if start_year > end_year:
        typer.echo(
            f"start-year ({start_year}) deve essere <= end-year ({end_year})",
            err=True,
        )
        raise typer.Exit(1)

    if scope_file:
        p = Path(scope_file)
        if not p.exists():
            typer.echo(f"scope-file not found: {p}", err=True)
            raise typer.Exit(1)
        try:
            with p.open() as f:
                scope = json.load(f)
        except json.JSONDecodeError as exc:
            typer.echo(f"scope-file is not valid JSON: {exc}", err=True)
            raise typer.Exit(1)
        if not isinstance(scope, list) or not all(isinstance(n, str) for n in scope):
            typer.echo("scope-file must contain a JSON array of strings", err=True)
            raise typer.Exit(1)
    else:
        parquet = CONFIG.data_processed / "matches.parquet"
        if not parquet.exists():
            typer.echo("matches.parquet non trovato — esegui `mondiali ingest` prima", err=True)
            raise typer.Exit(1)
        df = pd.read_parquet(parquet)
        df["date"] = pd.to_datetime(df["date"])
        scope = compute_tier3_scope(df)
        scope_out = CONFIG.data_processed / "tier3_scope.json"
        scope_out.parent.mkdir(parents=True, exist_ok=True)
        with scope_out.open("w") as f:
            json.dump(scope, f, indent=2)
        typer.echo(f"Computed scope: {len(scope)} nations → {scope_out}")

    cache_dir = CONFIG.data_raw / "transfermarkt" / "cache"
    output_path = CONFIG.data_raw / "transfermarkt" / "snapshots.parquet"
    years = list(range(start_year, end_year + 1))
    typer.echo(
        f"Scraping {len(scope)} nations × {len(years)} years = "
        f"{len(scope) * len(years)} target snapshots"
    )
    scrape_all(scope, years, cache_dir, output_path)
    typer.echo(f"Done. Output: {output_path}")


if __name__ == "__main__":
    app()
