"""Behavioural tests for v0 implementations."""

import numpy as np

import worldcup2026
from worldcup2026.evaluation.metrics import (
    brier_score,
    log_loss,
    ranked_probability_score,
)
from worldcup2026.models.dixon_coles import (
    DixonColesParams,
    btts_probability,
    fit,
    match_probabilities,
    match_rates,
    over_under_probability,
    score_matrix,
    time_decay_weights,
)
from worldcup2026.ratings.elo import (
    EloRatings,
    expected_score,
    goal_difference_multiplier,
)
from worldcup2026.simulation.tournament import (
    monte_carlo_group,
    sample_score_from_matrix,
    simulate_group,
)


def test_version():
    assert worldcup2026.__version__ == "0.0.1"


# --- Elo ---

def test_elo_expected_score_symmetric():
    assert expected_score(1500, 1500) == 0.5
    assert expected_score(1600, 1400) > 0.5
    assert expected_score(1400, 1600) < 0.5


def test_elo_is_zero_sum_per_match():
    r = EloRatings()
    r.update("A", "B", 2, 0)
    total = r.get("A") + r.get("B")
    assert abs(total - 2 * r.config.initial_rating) < 1e-9


def test_elo_winner_gains_rating():
    r = EloRatings()
    before = r.get("A")
    r.update("A", "B", 3, 0)
    assert r.get("A") > before
    assert r.get("B") < before


def test_goal_difference_multiplier_diminishing_returns():
    assert goal_difference_multiplier(1, 0) == 1.0
    assert goal_difference_multiplier(2, 0) == 1.5
    assert goal_difference_multiplier(5, 0) > goal_difference_multiplier(3, 0)


def test_elo_k_weights_ordered_by_importance():
    cfg = EloRatings().config
    assert (
        cfg.k_weights["world_cup"]
        > cfg.k_weights["confederation"]
        > cfg.k_weights["qualifier"]
        > cfg.k_weights["friendly"]
    )


def test_elo_update_batch_runs():
    r = EloRatings()
    r.update_batch(
        [
            {"home": "A", "away": "B", "home_goals": 1, "away_goals": 0},
            {"home": "B", "away": "C", "home_goals": 2, "away_goals": 2,
             "importance": "qualifier"},
            {"home": "A", "away": "C", "home_goals": 0, "away_goals": 1, "neutral": True},
        ]
    )
    ratings = r.as_dict()
    assert set(ratings) == {"A", "B", "C"}


# --- Dixon-Coles ---

def _equal_params(home_advantage: float = 0.0):
    return DixonColesParams(
        attack={"A": 0.2, "B": 0.2},
        defence={"A": 0.0, "B": 0.0},
        home_advantage=home_advantage,
        rho=0.0,
    )


def test_score_matrix_sums_to_one():
    m = score_matrix(1.4, 1.2, rho=-0.1)
    assert abs(m.sum() - 1.0) < 1e-9


def test_match_probabilities_sum_to_one():
    m = score_matrix(1.4, 1.2, rho=-0.1)
    assert abs(sum(match_probabilities(m)) - 1.0) < 1e-9


def test_equal_strength_symmetric_at_neutral():
    params = _equal_params()
    lam, mu = match_rates(params, "A", "B", neutral=True)
    p_h, _, p_a = match_probabilities(score_matrix(lam, mu, rho=0.0))
    assert abs(p_h - p_a) < 1e-9


def test_home_advantage_raises_home_win():
    p_neutral = _hwin(_equal_params(), neutral=True)
    p_home = _hwin(_equal_params(home_advantage=0.3), neutral=False)
    assert p_home > p_neutral


def _hwin(params, *, neutral):
    lam, mu = match_rates(params, "A", "B", neutral=neutral)
    return match_probabilities(score_matrix(lam, mu, rho=0.0))[0]


def test_dc_correction_raises_draw_prob():
    lam, mu = 1.2, 1.2
    _, p_d_indep, _ = match_probabilities(score_matrix(lam, mu, rho=0.0))
    _, p_d_dc, _ = match_probabilities(score_matrix(lam, mu, rho=-0.1))
    assert p_d_dc > p_d_indep


def test_over_under_partition_on_non_integer_line():
    m = score_matrix(1.6, 1.2, rho=-0.1)
    p_over, p_under = over_under_probability(m, line=2.5)
    assert abs(p_over + p_under - 1.0) < 1e-9


def test_btts_bounded():
    m = score_matrix(1.5, 1.2, rho=-0.1)
    assert 0.0 < btts_probability(m) < 1.0


# --- Dixon-Coles fit (synthetic round-trip) ---

def _generate_matches(true_params, teams, n_repeats, rng):
    matches = []
    for _ in range(n_repeats):
        for home in teams:
            for away in teams:
                if home == away:
                    continue
                lam, mu = match_rates(true_params, home, away)
                matrix = score_matrix(lam, mu, true_params.rho, max_goals=8)
                from worldcup2026.simulation.tournament import sample_score_from_matrix
                hg, ag = sample_score_from_matrix(matrix, rng)
                matches.append(
                    {"home": home, "away": away, "home_goals": hg, "away_goals": ag}
                )
    return matches


def test_fit_recovers_team_strength_ordering():
    teams = ["strong", "good", "mid", "weak", "minnow"]
    true_params = DixonColesParams(
        attack={"strong": 0.6, "good": 0.3, "mid": 0.0, "weak": -0.3, "minnow": -0.6},
        defence={"strong": -0.4, "good": -0.2, "mid": 0.0, "weak": 0.2, "minnow": 0.5},
        home_advantage=0.25,
        rho=-0.08,
    )
    rng = np.random.default_rng(42)
    matches = _generate_matches(true_params, teams, n_repeats=25, rng=rng)

    fitted = fit(matches, teams=teams)

    assert fitted.attack["strong"] > fitted.attack["mid"] > fitted.attack["minnow"]
    assert fitted.defence["strong"] < fitted.defence["mid"] < fitted.defence["minnow"]
    assert fitted.home_advantage > 0.0


def test_fit_centres_attack_to_zero_mean():
    teams = ["A", "B", "C"]
    true_params = DixonColesParams(
        attack={"A": 0.4, "B": 0.0, "C": -0.4},
        defence={"A": -0.2, "B": 0.0, "C": 0.2},
        home_advantage=0.2,
        rho=-0.05,
    )
    rng = np.random.default_rng(0)
    matches = _generate_matches(true_params, teams, n_repeats=30, rng=rng)

    fitted = fit(matches, teams=teams)
    mean_attack = sum(fitted.attack.values()) / len(teams)
    assert abs(mean_attack) < 1e-9


def test_time_decay_weights_recent_heavier_than_old():
    dates = ["2023-01-01", "2024-01-01", "2025-01-01"]
    w = time_decay_weights(dates, "2026-01-01", half_life_days=365.0)
    assert w[0] < w[1] < w[2]
    # halving every half-life — 1y old should be ~2x heavier than 2y old
    assert abs(w[2] / w[1] - 2.0) < 0.05


def test_fit_respects_time_decay():
    teams = ["A", "B"]
    true_old = DixonColesParams(
        attack={"A": 0.7, "B": -0.7}, defence={"A": -0.3, "B": 0.3},
        home_advantage=0.0, rho=0.0,
    )
    true_new = DixonColesParams(
        attack={"A": -0.7, "B": 0.7}, defence={"A": 0.3, "B": -0.3},
        home_advantage=0.0, rho=0.0,
    )
    rng = np.random.default_rng(1)
    old_matches = _generate_matches(true_old, teams, n_repeats=60, rng=rng)
    new_matches = _generate_matches(true_new, teams, n_repeats=60, rng=rng)
    matches = old_matches + new_matches
    weights = np.concatenate([np.full(len(old_matches), 0.01), np.full(len(new_matches), 1.0)])

    fitted = fit(matches, teams=teams, weights=weights)
    assert fitted.attack["B"] > fitted.attack["A"]


# --- Metrics ---

def test_log_loss_perfect_prediction_is_zero():
    probs = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert log_loss(probs, np.array([0, 1])) < 1e-10


def test_log_loss_uniform_equals_log_k():
    probs = np.full((4, 3), 1.0 / 3.0)
    assert abs(log_loss(probs, np.array([0, 1, 2, 1])) - np.log(3)) < 1e-9


def test_brier_perfect_is_zero():
    probs = np.array([[1.0, 0.0, 0.0]])
    assert brier_score(probs, np.array([0])) == 0.0


def test_rps_penalises_ordinal_distance():
    near = np.array([[0.0, 1.0, 0.0]])
    far = np.array([[0.0, 0.0, 1.0]])
    truth = np.array([0])
    assert ranked_probability_score(far, truth) > ranked_probability_score(near, truth)


# --- Simulation ---

def test_sample_score_deterministic_with_seed():
    m = score_matrix(1.5, 1.2, rho=-0.1)
    rng1 = np.random.default_rng(7)
    rng2 = np.random.default_rng(7)
    assert sample_score_from_matrix(m, rng1) == sample_score_from_matrix(m, rng2)


def test_simulate_group_ranks_dominant_team_first():
    teams = ["A", "B", "C", "D"]
    fixtures = [
        ("A", "B"), ("C", "D"),
        ("A", "C"), ("B", "D"),
        ("A", "D"), ("B", "C"),
    ]

    def sampler(home, away, _rng):
        if home == "A":
            return 2, 0
        if away == "A":
            return 0, 2
        return 1, 1

    standings = simulate_group(teams, fixtures, sampler, np.random.default_rng(0))
    assert standings[0].team == "A"
    assert standings[0].points == 9


def test_monte_carlo_group_probabilities_sum_to_one_per_team():
    teams = ["A", "B", "C", "D"]
    fixtures = [
        ("A", "B"), ("C", "D"),
        ("A", "C"), ("B", "D"),
        ("A", "D"), ("B", "C"),
    ]
    matrices = {
        ("A", "B"): score_matrix(2.0, 0.8), ("C", "D"): score_matrix(1.3, 1.3),
        ("A", "C"): score_matrix(1.8, 1.0), ("B", "D"): score_matrix(1.4, 1.2),
        ("A", "D"): score_matrix(2.1, 0.7), ("B", "C"): score_matrix(1.3, 1.3),
    }

    def sampler(home, away, rng):
        return sample_score_from_matrix(matrices[(home, away)], rng)

    probs = monte_carlo_group(teams, fixtures, sampler, n_runs=200, seed=1)
    for team_probs in probs.values():
        assert abs(sum(team_probs.values()) - 1.0) < 1e-9
