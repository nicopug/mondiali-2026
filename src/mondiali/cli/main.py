"""Entry point Typer CLI per il package `mondiali`.

Comandi disponibili in STEP 1:
    mondiali ingest        Download + parsing + Elo history -> matches.parquet
    mondiali baseline      Fit PriorBaseline su training set, report log-loss
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import structlog
import typer

from mondiali.config import CONFIG
from mondiali.data.ingestion import build_processed_matches, download_international_results
from mondiali.model.elo_logistic import EloLogisticBaseline
from mondiali.training.baseline_prior import PriorBaseline
from mondiali.training.evaluate import log_loss_1x2

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


if __name__ == "__main__":
    app()
