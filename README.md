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

## Real data: re-simulate from the current state

The pipeline also runs on real data. The
[`martj42/international_results`](https://github.com/martj42/international_results)
dataset is a single CSV of every men's international since 1872 — and it now
carries the WC2026 fixture list too, with unplayed matches scored `NA`. So one
download gives us both the training history and the live tournament state.

```bash
# 1. Download the dataset (gitignored — lands in data/raw/)
curl -sSL -o data/raw/international_results.csv \
  https://raw.githubusercontent.com/martj42/international_results/master/results.csv

# 2. Fit on history, condition on results played so far, and re-simulate the rest
python scripts/resim_current_state.py            # 10k sims, history since 2015
python scripts/resim_current_state.py --n-runs 20000 --since 2018-01-01
```

The script fits Dixon-Coles with time-decay weighting and neutral-venue
handling, derives the 12 groups from the fixtures, fixes the already-played
group results via the simulator's `known_results` hook, and prints updated
champion / round-reached / group-advancement probabilities. The full 48-team
table is written to `data/processed/wc2026_resim_<date>.csv`.

The knockout stage uses FIFA's **fixed** bracket — group winners/runners-up in
their published R32 slots, third-placed teams routed under Annex C's group
constraints — rather than a generic seeded bracket (see `simulation/bracket.py`).

Conditioning a simulation on matches already played is a first-class feature of
the engine — see `known_results` in `simulation/tournament.py`.

**Host residual.** World Cup venues are treated as neutral, except the three
co-hosts (Mexico, USA, Canada) carry a small `--host-advantage` log-rate edge
whenever they play (default 0.15; pass `0` to disable).

## Tuning the fit window

The training window and time-decay half-life are tuned by backtesting 1X2
log-loss / RPS on two holdouts (the WC2022 finals, and recent competitive
internationals):

```bash
python scripts/backtest_window.py     # grid over window-years x half-life
```

It prints a ranked table and the recommended `(window, half-life)`; those
values feed `config/default.yaml` (`world_cup.fit_window_years` /
`fit_half_life_days`) and the `resim_current_state.py` defaults.

## Betting layer (MVP in progress)

The project is now building toward a betting MVP — see [ROADMAP.md](ROADMAP.md)
for the sequenced plan and the edge thesis (*calibrate marginals to the market,
win on correlation, measure by CLV*). Landed so far:

- `data/odds.py` — bookmaker odds ingestion into one normalized table (The Odds
  API; key via `.env` `ODDS_API_KEY`; works offline from cached snapshots).
- `betting/clv.py` — closing-line-value & P&L harness: CLV-EV vs the no-vig
  close, beat-close rate, settlement, and an aggregate report. CLV is the
  go/no-go signal for whether the model has real edge.
- `betting/markets.py` — 1X2, double chance, totals, BTTS and **same-game
  multis** read off the joint score matrix (true correlation, not naive
  leg-multiplication).
- `betting/blend.py` — anchor a match's marginals (1X2/totals) to vig-removed
  market prices while keeping the model's correlation structure (IPF).
- `betting/acca.py` — build accumulators across different fixtures.
- `betting/edge.py` — model vs market → EV filter, fractional Kelly, exposure
  caps, bet-sheet summary and a CLV bet log.

- `evaluation/calibration.py` — temperature scaling + reliability/ECE. The
  model is under-confident; `scripts/calibrate.py` fits the correction and
  re-verifies it against the market. Pass `--temperature` to the scripts below.

```bash
python scripts/clv_demo.py        # CLV/ROI report on a synthetic bet ledger
python scripts/calibrate.py       # fit calibration temperature, verify vs market
python scripts/build_accas.py     # high-accuracy 2-3 game accas from the model
python scripts/find_value.py --demo   # +EV staked bet sheet (synthetic market)
# python scripts/find_value.py --odds data/raw/odds/odds_*.csv --temperature 0.914
```

> **Reality check (Gate G1):** against a 25-book consensus the model has **no
> tradeable single-1X2 edge** — see `ROADMAP.md`. The model is sound but
> under-confident; the live edge has to come from correlation in same-game
> multis vs *soft* books, not from out-predicting sharp 1X2 lines.

Knockout ties are decided by a strength-weighted penalty shootout (mild logistic
tilt) rather than a coin flip — see `shootout_p` in `simulation/tournament.py`.

## Contributing / development notes

- `src/worldcup2026/` is the package; everything is importable as
  `from worldcup2026.<module> import ...`
- Real data files belong in `data/raw/` and `data/processed/` — both gitignored.
  Never commit raw match data or scraped datasets unless the licence permits.
- Secrets (API keys for odds providers, etc.) go in `.env`, which is gitignored.
  `.env.example` is the safe-to-commit template.

## Licence

TBD.
