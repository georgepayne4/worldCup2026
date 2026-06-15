# World Cup 2026 — Execution Roadmap (MVP-first)

> Sequenced plan for the remaining work, to execute against one milestone at a
> time. Complements [METHODOLOGY.md](METHODOLOGY.md) (the *what/why*); this is
> the *in-what-order*. Re-ordered to reach an **effective MVP** — a thing that
> emits staked, edge-measured bet recommendations — before breadth.

**Status:** core model is built and conditioned on live results (sessions 1–3).
But there is **no evidence of edge yet**: every metric so far is model-vs-reality,
never model-vs-*market*, and on tournament games the model is only ~level with a
naive 1/3 baseline. The MVP exists to confront that head-on.

---

## The edge thesis (read this first)

The goal is **not** to out-predict sharp bookmakers on 1X2 / outrights — that
market is brutally efficient and we lack the data/capital to win it. The goal is
a **structural** edge:

1. **Calibrate marginals to the market.** The closing line is the best single
   predictor in football. We pull our 1X2/totals toward vig-removed market
   prices, so our marginals are at least market-accurate.
2. **Win on correlation.** Soft books price same-game-multi legs as if they were
   independent. Our Dixon-Coles `score_matrix` is the *exact joint distribution*
   of a match, so we price "Team A win AND over 2.5 AND BTTS" with the true
   correlation. **We don't need to beat the legs — only the book's correlation
   assumption.** This is the highest-value, most achievable edge in the project.
3. **Measure everything by CLV.** Closing-line value (did our price beat the
   eventual close?) is the only leading indicator of long-run profit. P&L is
   noise over small samples; CLV is signal.

**Honest ceiling.** This can credibly reach "*small, real, measurable edge in
correlated soft-book markets, staked with discipline*" — a legitimate semi-pro
tier. Starlizard-grade (player-level/xG data, in-play, proprietary feeds,
massive line-shopping) is out of scope; we deliberately attack the niche a small
operation *can* win.

---

## MVP — prove or kill the edge (3 milestones)

### MVP-1 — Market data + CLV measurement  ✅ *done*

**Goal.** Be able to *measure* edge. Without prices and a CLV harness, nothing
else is verifiable.

**Build**
- `data/odds.py` — normalized odds schema; The Odds API client (key-guarded)
  with timestamped snapshots to `data/raw/odds/` (gitignored); snapshot loader.
- `betting/clv.py` — no-vig closing probabilities, **CLV-EV** (EV at your price
  using the no-vig close as truth), beat-the-close rate, bet settlement, and an
  aggregate backtest report (n, ROI, mean CLV, beat-close %).
- Demo + tests on a synthetic odds/bets/results fixture (live history needs a
  paid feed or forward-collected snapshots — the harness is provider-agnostic).

**Acceptance.** CLV math unit-tested; demo prints a CLV/ROI report end-to-end;
odds loader round-trips a snapshot; suite + ruff green.

### MVP-2 — Edge engine (market-blend + correlated markets)  ✅ *done*

**Goal.** Produce prices where we have structural edge.

**Built**
- `betting/blend.py` — IPF (Sinkhorn) blend of the joint `score_matrix`'s 1X2
  and/or totals marginals toward vig-removed market prices, preserving the
  model's correlation structure.
- `betting/markets.py` — 1X2, double chance, totals (any line), BTTS read off
  the (blended) joint, plus **same-game multi pricing** that exposes the
  correlation premium vs a leg-multiplying book.
- `betting/acca.py` + `scripts/build_accas.py` — cross-game accumulator builder
  (independent legs, one per match; accuracy/value modes) that suggests
  placeable 2–3 game accas from the most confident model selections.

**Done when** (all met): markets partition correctly; a correlated multi prices
away from the naive product in the right direction; blended marginals match
supplied market prices to tolerance; acca combines legs with correct
prob/odds/EV. (`tests/test_markets.py`.)

### MVP-3 — Staking, recommendations + cheap accuracy fix  ✅ *done*

**Goal.** Emit a disciplined, staked bet sheet end-to-end.

**Built**
- `betting/edge.py` — model vs market → EV filter, **fractional** Kelly,
  per-bet / per-match / total exposure caps, bet-sheet summary + CLV log.
- `scripts/find_value.py` — fit → price remaining fixtures → join market odds
  (`--odds` snapshot or `--demo`) → +EV bet sheet with stakes.
- **strength-weighted shootouts** — `simulate_knockout_match` now takes a
  `shootout_p` (logistic on net strength), wired through the sim/MC and resim;
  fixes the coin-flip bias in every outright/stage price.

**Deferred** (moved to Post-MVP P3, needs a data source):
- **squad-value covariate** (Transfermarkt national-team value) — would be the
  80/20 of "player quality" for thin-history sides, but requires scraping; it
  belongs with the player-data milestone, not the no-new-data MVP.

**Done when** (all met): bet sheet lists +EV bets with fair vs offered odds,
EV%, Kelly stake, exposure within caps; bets logged for CLV; weighted shootouts
tested; suite + ruff green. (`tests/test_edge.py`, `tests/test_smoke.py`.)

### Gate G1 — after MVP: *is there measurable edge?*
Run the MVP forward/back on real prices. If we don't see positive CLV on the
correlated markets, **stop staking** and return to core hardening (Post-MVP P1)
before risking money. This gate decides whether the rest of the roadmap is worth
building.

**Result (June 2026, real Odds API snapshot).** Model 1X2 correlates 0.96 with a
25-book consensus (MAE 5pts) but is **under-confident** — it shaves favourites
(~−4.5pts) and inflates underdogs (~+2.2pts), so single-bet "value" is
concentrated on longshots and is *miscalibration, not edge*. Verdict: **no
tradeable single-1X2 edge; do not stake the raw sheet.** P1 calibration
(temperature 0.914) halves the average bias and improves holdout log-loss/ECE,
but does **not** fix large per-match disagreements with the market. Conclusion
holds: don't bet single 1X2 vs sharp books — pursue **blend-to-market marginals
+ correlation edge in same-game multis vs *soft* books**.

---

## Post-MVP (sequenced, after G1)

| # | Milestone | Depends on | Why here |
|---|-----------|-----------|----------|
| P1 | **Core hardening & uncertainty** — ✅ calibration (temperature + reliability/ECE, `evaluation/calibration.py`, `scripts/calibrate.py`) done; **remaining:** blend-to-market in the value pipeline, Elo→DC seeding with confederation shrinkage, **Bayesian/partial-pooling** strengths so Kelly uses *uncertainty* not point estimates | G1 | Bigger, only worth it once edge is shown; de-risks staking |
| P2 | **Tournament markets** — to-win / reach-stage / group-winner / to-qualify, priced from existing MC outputs | MVP-2 | Cheap (sim already produces these); sharper market, lower edge |
| P3 | **Player/squad data** — start with the **squad-value covariate** (cheap accuracy win deferred from MVP-3), then minutes/involvement | G2 | Largest external dependency; the gateway to props |
| P4 | **Player props** — anytime/2+/first scorer, assists, shots, cards, golden boot | P3, MVP-2 | Soft books are weakest here — high edge, high data cost |
| P5 | **In-tournament Bayesian updating** — strengths adapt as matches are played | P1 | The "learning model" (METHODOLOGY §9) |
| P6 | **Serving, performance & cross-checks** — Streamlit/CLI bet dashboard, polars/joblib MC, optional PyMC fit | most | Make it usable/fast once it's worth using |

### Gate G2 — before P3: *player-data viability.* Confirm a licence-clean,
sufficiently complete player-data source exists; if not, defer P3–P4 and do P5.

---

## Backlog (low coupling, pick up anytime)

- Embed the exact 495-row Annex C third-place table (replace constrained matching).
- Strength-weighted tiebreakers (H2H / fair-play / lots) in `simulate_group`.
- Fatter-tailed / negative-binomial goals if correct-score backtests warrant.
- CI (GitHub Actions: pytest + ruff).
- In-play / live re-pricing (re-sim from current score & clock).
- Data-refresh automation as the tournament progresses.
