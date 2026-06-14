# World Cup 2026 — Modelling Methodology

> **Status:** v0 draft. This document is the starting blueprint for the project. Numbers,
> models, and data sources will be refined as we backtest and iterate.

---

## 1. Objective

Build a probabilistic model of the 2026 FIFA World Cup (USA / Canada / Mexico, 48 teams,
expanded format) that produces:

1. **Match-level probabilities** — P(home win), P(draw), P(away win), and an expected
   goals distribution for every fixture.
2. **Tournament-level probabilities** — for every team, the probability of topping their
   group, advancing past each knockout round, and lifting the trophy.
3. **A reusable simulation engine** — so that "what if X is injured" / "what if group Y
   reshuffles" questions can be answered quickly.

The downstream goal (see §9) is to extend this into a **betting-market model** that finds
value beyond the obvious favourites — specifically in bet builders, props, and correlated
multi-leg markets.

---

## 2. Why this approach

Most public World Cup models fall into one of three buckets:

| Approach | Strength | Weakness |
|----------|----------|----------|
| Pure Elo / rating systems | Simple, well-calibrated for binary outcomes | Doesn't model goals, can't price totals/handicaps |
| Bookmaker-implied probabilities | Sharp, well-calibrated | No edge — you are the market |
| Goal-based statistical models (Poisson, Dixon-Coles) | Models the full score distribution | Needs careful calibration; vanilla Poisson under-prices draws |

We combine **(1) a team-strength layer** (Elo-style rating, tuned on international fixtures
and adjusted for tournament context) with **(2) a goal-generating layer** (Dixon-Coles
bivariate Poisson, parameterised by the strength ratings). The team-strength layer gives
us robust win/draw/loss probabilities; the goal layer gives us everything else (totals,
both-teams-to-score, correct score, handicaps) — which is the foundation we need for the
betting extension.

---

## 3. Theoretical foundations

### 3.1 Team strength — Elo with international adjustments

Standard Elo with the following adjustments used by the World Football Elo Ratings and
academic literature:

- **Match importance weight** `K` scales by competition: friendlies < qualifiers <
  confederation championships < World Cup.
- **Goal difference multiplier** — wins by larger margins move ratings further, with a
  diminishing-returns curve so 8–0 doesn't dominate.
- **Home advantage** — fixed point adjustment, but **set to zero for the World Cup**
  (neutral venues), with a small residual for CONCACAF teams playing in NA.
- **Confederation strength priors** — a Bayesian prior pulls under-observed teams
  (especially smaller AFC / CAF nations) toward their confederation mean until enough
  matches accumulate.

### 3.2 Match outcomes — Dixon-Coles bivariate Poisson

For a fixture between teams $i$ (home) and $j$ (away), model goals $(X, Y)$ as:

$$
\lambda_i = \exp(\alpha_i + \beta_j + \gamma_h),\quad
\mu_j     = \exp(\alpha_j + \beta_i)
$$

where $\alpha$ is attacking strength, $\beta$ is defensive weakness, and $\gamma_h$ is
home advantage. The **Dixon-Coles correction** $\tau(x, y, \lambda, \mu, \rho)$ adjusts
the joint probability for low-scoring outcomes (0-0, 1-0, 0-1, 1-1) to fix vanilla
Poisson's well-known under-prediction of draws.

Strength parameters $\alpha_i, \beta_i$ are seeded from the Elo layer and then re-fit by
maximum likelihood with **exponential time-decay weighting** (half-life ~18 months) so
recent form dominates.

### 3.3 Tournament progression — Monte Carlo

Group stage and knockout bracket are simulated $N \geq 20{,}000$ times. Each simulation:

1. Sample every group-stage match score from the bivariate Poisson.
2. Apply FIFA tiebreakers (points → GD → GF → head-to-head → fair play → draw of lots).
3. Resolve the 48-team expanded knockout bracket (32-team round of 32 → R16 → QF → SF →
   F) using FIFA's **fixed** published bracket: each group winner / runner-up occupies
   its pre-assigned R32 slot and winners feed a fixed tree (matches 73–104). The eight
   best third-placed teams are routed into their R32 slots under Annex C's constraints —
   each slot only accepts thirds from a published set of groups, and no team meets its
   own group (see `simulation/bracket.py`). Extra time and penalties are sampled when
   needed.
4. Record bracket outcomes per team.

Outputs are reported as Monte Carlo probability estimates with binomial confidence
intervals.

---

## 4. Data sources

| Source | Use | Notes |
|--------|-----|-------|
| International match results (e.g. Kaggle `martj42/international_football_results`) | Elo + Dixon-Coles fit | Public, long history, needs cleaning |
| FIFA / Elo rankings | Prior + cross-check | Read-only signal |
| Squad data (FBref, Transfermarkt) | Player-availability adjustment | Manual scraping; respect terms of use |
| Fixture list (official FIFA 2026 schedule) | Tournament structure | Locked in by group draw |
| Bookmaker odds (Pinnacle / Betfair Exchange snapshots) | Calibration target + market-edge measurement | For evaluation only; not a training input |

Raw data lives under `data/raw/` (gitignored), processed parquet under `data/processed/`.

---

## 5. Implementation stack

- **Language:** Python 3.11+
- **Core libs:** `numpy`, `pandas` (or `polars` for speed), `scipy`, `statsmodels`
- **Optimisation / fitting:** `scipy.optimize` for MLE; optionally `PyMC` for a Bayesian
  hierarchical extension
- **Simulation:** vectorised NumPy; `joblib` for embarrassingly-parallel MC runs
- **Validation:** `scikit-learn` metrics, custom calibration plots
- **Notebooks:** `jupyter` for exploration; production code lives in `src/`

---

## 6. Evaluation & calibration

A model that *looks* sharp but is poorly calibrated is useless for betting. Backtest
protocol:

1. **Holdout tournaments** — refit using only data available before kickoff of WC 2018
   and WC 2022, then score predictions against actual outcomes.
2. **Metrics:**
   - Log loss / Brier score on 1X2 outcomes
   - Ranked probability score (RPS) — penalises bigger errors in ordinal outcomes
   - Reliability diagrams across probability buckets
   - For totals / BTTS: log loss vs. closing Pinnacle line (the sharpest public benchmark)
3. **Sanity checks:** simulated group-stage point distributions should match historical
   distributions; champion probabilities should sum to 1.

A target: **beat closing Pinnacle on 1X2 by ≥1% in log loss** across the backtest. If we
can't, we don't have an edge and the betting extension is dead on arrival.

---

## 7. Project structure

```
src/worldcup2026/
├── data/         # ingestion, cleaning, feature engineering
├── ratings/      # Elo + adjustments
├── models/       # Dixon-Coles & extensions
├── simulation/   # Monte Carlo tournament engine
├── evaluation/   # backtesting, calibration, metrics
└── betting/      # (future) market modelling — see §9
```

---

## 8. Assumptions & limitations

- **Neutral-venue assumption** — home advantage is zeroed for the WC. The three host
  nations get a small residual bonus; this may under-state it.
- **Injuries / squad changes** are not modelled in v0. We treat team strength as
  team-level. Player-level adjustments come later.
- **Tactical match-ups** (e.g. high-press vs. low-block) are absorbed into the noise term.
  No style-vs-style interaction in v0.
- **Refereeing, weather, travel fatigue** — not modelled.
- **Sample-size cliff** — small CAF / OFC sides have thin international match histories;
  the confederation prior partially mitigates this but is itself a strong assumption.

---

## 9. Future work — betting-market extension

> **Note for next phase.** Once the core methodology backtests cleanly, the project
> expands into a **betting prediction layer**. The goal is **not** to pick favourites
> (the market already does that well) but to find mispriced complex markets, especially:

- **Bet builders / same-game multis** — pricing correlated outcomes (e.g. "Team A wins
  AND over 2.5 goals AND a player scores"). Bookmakers often under-price negative
  correlation and over-price positive correlation. Our Poisson framework gives us joint
  distributions to exploit this directly.
- **Player props** — goals, assists, shots, cards. Requires a player-level minutes &
  involvement model layered on top of team strength.
- **Tournament markets** — top scorer, golden ball, stage of elimination per team.
- **Live / in-play hooks** — re-running simulations from current score & time remaining.
- **Pattern learning from the tournament itself** — once group-stage matches are played,
  Bayesian updating of strength parameters (and detection of "tournament form" effects:
  some teams historically peak in knockouts). This is where a learning model — not just
  a fixed pre-tournament model — earns its keep.
- **Market-edge filter** — every recommended bet must clear a configurable EV threshold
  vs. the best available price across a tracked set of bookmakers, and must pass a
  Kelly-fractional staking sanity check.

The betting layer will live in `src/worldcup2026/betting/` and will be developed only
after the core match-prediction and simulation modules are calibrated against historical
tournaments. **No betting code is wired up in v0.**

---

## 10. Open questions for the next session

These are deliberately left open for the next review:

1. **Polars vs. pandas** — worth the cognitive overhead at this scale?
2. **Bayesian PyMC version** — implement alongside MLE as a calibration cross-check, or
   defer?
3. **Live odds data** — which provider for backtesting (Betfair historical, OddsPortal
   scraping, paid API)?
4. **Player-level data** — start gathering now (long lead time) or wait for v1?
5. **Hosting / serving** — is the end-state a notebook, a CLI, a Streamlit dashboard, or
   an API? This shapes the structure.
