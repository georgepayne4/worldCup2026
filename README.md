# worldCup2026

Probabilistic modelling of the 2026 FIFA World Cup — match outcomes, tournament
progression, and (later) betting-market analysis.

> **Status:** v0 scaffolding. See [METHODOLOGY.md](METHODOLOGY.md) for the full plan.

---

## What this is

A Python project that:

1. Fits team-strength ratings from international match history.
2. Models each fixture as a Dixon-Coles bivariate Poisson to get full score
   distributions (not just W/D/L).
3. Runs a Monte Carlo simulation of the 48-team tournament to produce per-team
   probabilities at every stage.
4. **Future:** extends to a betting-market layer focused on bet builders, props,
   and correlated multi-leg markets — see §9 of `METHODOLOGY.md`.

## Project layout

```
.
├── METHODOLOGY.md           # the plan — read this first
├── src/worldcup2026/
│   ├── data/                # ingestion, cleaning, features
│   ├── ratings/             # Elo + international adjustments
│   ├── models/              # Dixon-Coles & extensions
│   ├── simulation/          # Monte Carlo tournament engine
│   ├── evaluation/          # backtesting, calibration, metrics
│   └── betting/             # (future) market modelling — empty in v0
├── tests/
├── notebooks/               # exploratory analysis
├── data/                    # raw/ and processed/ — gitignored
├── config/                  # configuration files
├── pyproject.toml
├── requirements.txt
└── .env.example             # copy to .env (gitignored) for any secrets
```

## Quickstart

```bash
# 1. Clone
git clone https://github.com/georgepayne4/worldCup2026.git
cd worldCup2026

# 2. Create a virtual env
python -m venv .venv
source .venv/Scripts/activate    # Windows bash
# or: .venv\Scripts\activate     # Windows cmd/PowerShell

# 3. Install
pip install -e ".[dev]"

# 4. Copy environment template
cp .env.example .env             # then fill in any API keys locally

# 5. Run tests
pytest

# 6. Run the end-to-end demo (synthetic data, ~10s)
python scripts/example.py
```

The demo generates a synthetic match history, fits Dixon-Coles, and runs 500
Monte Carlo World Cups — printing top-10 champion / final / SF / QF
probabilities. It's the cleanest illustration of how the modules compose.

## Contributing / development notes

- `src/worldcup2026/` is the package; everything is importable as
  `from worldcup2026.<module> import ...`
- Real data files belong in `data/raw/` and `data/processed/` — both gitignored.
  Never commit raw match data or scraped datasets unless the licence permits.
- Secrets (API keys for odds providers, etc.) go in `.env`, which is gitignored.
  `.env.example` is the safe-to-commit template.

## Licence

TBD.
