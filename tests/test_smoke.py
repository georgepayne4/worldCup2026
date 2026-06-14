"""Behavioural tests for v0 implementations."""

from itertools import combinations

import numpy as np
import pytest

import worldcup2026
from worldcup2026.betting.odds import (
    expected_value,
    kelly_fraction,
    odds_to_probability,
    probability_to_odds,
    remove_vig,
)
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
    predict_match,
    score_matrix,
    time_decay_weights,
)
from worldcup2026.ratings.elo import (
    EloRatings,
    expected_score,
    goal_difference_multiplier,
)
from worldcup2026.simulation.bracket import (
    GROUP_LETTERS,
    KNOCKOUT_TREE,
    R32_LEAF_ORDER,
    R32_MATCHES,
    THIRD_SLOT_GROUPS,
    assign_thirds,
    bracket_order,
)
from worldcup2026.simulation.tournament import (
    cumulative_round_probabilities,
    monte_carlo_group,
    monte_carlo_world_cup,
    sample_score_from_matrix,
    simulate_group,
    simulate_knockout_match,
    simulate_world_cup,
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


def test_predict_match_returns_consistent_summary():
    params = DixonColesParams(
        attack={"A": 0.3, "B": -0.2}, defence={"A": -0.1, "B": 0.1},
        home_advantage=0.2, rho=-0.05,
    )
    p = predict_match(params, "A", "B")
    assert p.home_team == "A" and p.away_team == "B"
    assert abs(p.p_home_win + p.p_draw + p.p_away_win - 1.0) < 1e-9
    assert abs(p.p_over_2_5 + p.p_under_2_5 - 1.0) < 1e-9
    assert 0.0 < p.p_btts < 1.0
    assert p.expected_home_goals > p.expected_away_goals
    assert p.p_home_win > p.p_away_win


def test_match_rates_host_boost_lifts_only_that_side():
    params = DixonColesParams(
        attack={"A": 0.0, "B": 0.0}, defence={"A": 0.0, "B": 0.0},
        home_advantage=0.0, rho=0.0,
    )
    lam0, mu0 = match_rates(params, "A", "B", neutral=True)
    lam_h, mu_h = match_rates(params, "A", "B", neutral=True, home_boost=0.2)
    assert lam_h > lam0 and mu_h == mu0
    lam_a, mu_a = match_rates(params, "A", "B", neutral=True, away_boost=0.2)
    assert mu_a > mu0 and lam_a == lam0


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


# --- Betting primitives ---

def test_odds_probability_roundtrip():
    assert abs(probability_to_odds(0.5) - 2.0) < 1e-12
    assert abs(odds_to_probability(2.0) - 0.5) < 1e-12


def test_remove_vig_normalises_and_preserves_ordering():
    fair = remove_vig([1.9, 3.5, 4.5])
    assert abs(sum(fair) - 1.0) < 1e-9
    assert fair[0] > fair[1] > fair[2]


def test_expected_value_zero_at_fair_odds():
    assert abs(expected_value(0.5, 2.0)) < 1e-12
    assert expected_value(0.6, 2.0) > 0.0
    assert expected_value(0.4, 2.0) < 0.0


def test_kelly_zero_when_negative_ev():
    assert kelly_fraction(0.4, 2.0) == 0.0
    assert 0.0 < kelly_fraction(0.6, 2.0) <= 1.0


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


def test_simulate_group_uses_known_results_and_skips_sampling():
    teams = ["A", "B", "C", "D"]
    fixtures = [
        ("A", "B"), ("C", "D"),
        ("A", "C"), ("B", "D"),
        ("A", "D"), ("B", "C"),
    ]
    # Every fixture is already played -> the sampler must never be called and the
    # standings are fully determined by the observed scores.
    known = {
        ("A", "B"): (3, 0), ("C", "D"): (0, 0),
        ("A", "C"): (2, 0), ("B", "D"): (1, 1),
        ("A", "D"): (1, 0), ("B", "C"): (0, 0),
    }

    def sampler(_h, _a, _r):
        raise AssertionError("sampler called for an already-played fixture")

    standings = simulate_group(
        teams, fixtures, sampler, np.random.default_rng(0), known_results=known
    )
    by_team = {s.team: s for s in standings}
    assert standings[0].team == "A"
    assert by_team["A"].points == 9 and by_team["A"].goals_for == 6
    assert by_team["A"].goals_against == 0


def test_known_results_partial_mix_played_and_sampled():
    teams = ["A", "B", "C", "D"]
    fixtures = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D"), ("B", "C")]
    known = {("A", "B"): (5, 0)}  # only this one is played

    def sampler(_h, _a, _r):
        return 0, 0  # everything else is a goalless draw

    standings = simulate_group(
        teams, fixtures, sampler, np.random.default_rng(0), known_results=known
    )
    by_team = {s.team: s for s in standings}
    # A won its only decisive game 5-0; everyone else drew everything.
    assert by_team["A"].goals_for == 5 and by_team["A"].won == 1
    assert by_team["B"].goals_against == 5


def test_monte_carlo_world_cup_respects_known_results():
    groups = _make_groups_48()
    # Force T00 to have thrashed its three group rivals already; with a coin-flip
    # sampler elsewhere it should still reach the knockout in every run.
    g_a = groups["A"]
    known = {
        (g_a[0], g_a[1]): (5, 0),
        (g_a[0], g_a[2]): (5, 0),
        (g_a[0], g_a[3]): (5, 0),
    }

    def sampler(_h, _a, r):
        return int(r.integers(0, 2)), int(r.integers(0, 2))

    probs = monte_carlo_world_cup(
        groups, sampler, n_runs=50, seed=4,
        fixtures_fn=lambda ts: [
            (ts[0], ts[1]), (ts[0], ts[2]), (ts[0], ts[3]),
            (ts[1], ts[2]), (ts[1], ts[3]), (ts[2], ts[3]),
        ],
        known_results=known,
    )
    assert probs[g_a[0]]["group_stage"] == 0.0  # never eliminated in the group


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


# --- Full 48-team tournament ---

def _make_groups_48():
    teams = [f"T{i:02d}" for i in range(48)]
    return {chr(65 + g): teams[g * 4 : (g + 1) * 4] for g in range(12)}


# --- FIFA fixed knockout bracket ---

def test_leaf_order_folds_into_official_tree():
    # Pairwise-folding R32 winners in leaf order must reproduce KNOCKOUT_TREE.
    current = list(R32_LEAF_ORDER)
    expected_round_roots = [
        [89, 90, 91, 92, 93, 94, 95, 96],
        [97, 98, 99, 100],
        [101, 102],
        [104],
    ]
    for roots in expected_round_roots:
        nxt = []
        for i in range(0, len(current), 2):
            a, b = current[i], current[i + 1]
            # find the match whose two feeders are {a, b}
            match = next(m for m, fb in KNOCKOUT_TREE.items() if set(fb) == {a, b})
            nxt.append(match)
        assert sorted(nxt) == roots
        current = nxt
    assert current == [104]
    assert len(R32_LEAF_ORDER) == 16 and len(set(R32_LEAF_ORDER)) == 16


def test_assign_thirds_valid_for_all_495_combinations():
    for combo in combinations(GROUP_LETTERS, 8):
        assignment = assign_thirds(list(combo))
        # one team per third slot, exactly the qualifying groups, each in-range
        assert set(assignment) == set(THIRD_SLOT_GROUPS)
        assert sorted(assignment.values()) == sorted(combo)
        for match, group in assignment.items():
            assert group in THIRD_SLOT_GROUPS[match]
            winner_slot = R32_MATCHES[match][0]
            assert winner_slot[0] == "W" and winner_slot[1] != group  # no self-meet


def test_assign_thirds_rejects_wrong_count():
    with pytest.raises(ValueError):
        assign_thirds(list("ABCDEFG"))  # only 7


def test_bracket_order_never_pairs_same_group_in_r32():
    winners = {g: f"{g}1" for g in GROUP_LETTERS}
    runners_up = {g: f"{g}2" for g in GROUP_LETTERS}
    qualifying = list("ABCDEFGH")  # any 8 groups
    thirds = {g: f"{g}3" for g in qualifying}
    order = bracket_order(winners, runners_up, thirds, assign_thirds(qualifying))
    assert len(order) == 32 and len(set(order)) == 32
    for i in range(0, 32, 2):
        assert order[i][0] != order[i + 1][0]  # group letter prefix differs


def test_simulate_world_cup_requires_groups_a_to_l():
    bad_groups = {str(i): [f"x{i}{j}" for j in range(4)] for i in range(12)}

    def sampler(_h, _a, r):
        return int(r.integers(0, 3)), int(r.integers(0, 3))

    with pytest.raises(ValueError):
        simulate_world_cup(bad_groups, sampler, np.random.default_rng(0))


def test_simulate_world_cup_round_counts_invariant():
    groups = _make_groups_48()

    def sampler(_h, _a, r):
        return int(r.integers(0, 4)), int(r.integers(0, 4))

    reached = simulate_world_cup(groups, sampler, np.random.default_rng(0))
    counts = {label: sum(1 for v in reached.values() if v == label) for label in {
        "group_stage", "round_of_32", "round_of_16",
        "quarter_final", "semi_final", "final", "champion",
    }}
    assert counts == {
        "group_stage": 16, "round_of_32": 16, "round_of_16": 8,
        "quarter_final": 4, "semi_final": 2, "final": 1, "champion": 1,
    }


def test_dominant_team_always_wins_the_tournament():
    groups = _make_groups_48()

    def sampler(home, away, _r):
        if home == "T00":
            return 5, 0
        if away == "T00":
            return 0, 5
        return 0, 0

    reached = simulate_world_cup(groups, sampler, np.random.default_rng(7))
    assert reached["T00"] == "champion"


def test_cumulative_round_probabilities_monotone_and_bounded():
    fake = {
        "X": {
            "group_stage": 0.40, "round_of_32": 0.30, "round_of_16": 0.15,
            "quarter_final": 0.08, "semi_final": 0.04, "final": 0.02, "champion": 0.01,
        }
    }
    cum = cumulative_round_probabilities(fake)
    assert abs(cum["X"]["group_stage"] - 1.0) < 1e-9
    assert abs(cum["X"]["champion"] - 0.01) < 1e-9
    assert abs(cum["X"]["final"] - 0.03) < 1e-9
    order = ["group_stage", "round_of_32", "round_of_16", "quarter_final",
             "semi_final", "final", "champion"]
    vals = [cum["X"][r] for r in order]
    assert all(vals[i] >= vals[i + 1] for i in range(len(vals) - 1))


def test_monte_carlo_world_cup_probabilities_sum_to_one_per_team():
    groups = _make_groups_48()

    def sampler(_h, _a, r):
        return int(r.integers(0, 3)), int(r.integers(0, 3))

    probs = monte_carlo_world_cup(groups, sampler, n_runs=15, seed=2)
    for team_probs in probs.values():
        assert abs(sum(team_probs.values()) - 1.0) < 1e-9


def test_et_sampler_used_only_on_drawn_90():
    et_count = {"n": 0}

    def main_draw(_h, _a, _r):
        return 1, 1

    def main_win(_h, _a, _r):
        return 2, 0

    def et(_h, _a, _r):
        et_count["n"] += 1
        return 3, 0

    rng = np.random.default_rng(0)
    winner, _ = simulate_knockout_match("A", "B", main_draw, rng, et_sampler=et)
    assert et_count["n"] == 1 and winner == "A"

    et_count["n"] = 0
    winner, _ = simulate_knockout_match("A", "B", main_win, rng, et_sampler=et)
    assert et_count["n"] == 0 and winner == "A"


def test_et_drawn_falls_through_to_coin_flip():
    et_count = {"n": 0}

    def main_draw(_h, _a, _r):
        return 0, 0

    def et_draw(_h, _a, _r):
        et_count["n"] += 1
        return 1, 1

    rng = np.random.default_rng(0)
    winner, loser = simulate_knockout_match("A", "B", main_draw, rng, et_sampler=et_draw)
    assert et_count["n"] == 1
    assert {winner, loser} == {"A", "B"}


def test_monte_carlo_world_cup_dominant_team_wins_every_run():
    groups = _make_groups_48()

    def sampler(home, away, _r):
        if home == "T00":
            return 4, 0
        if away == "T00":
            return 0, 4
        return 0, 0

    probs = monte_carlo_world_cup(groups, sampler, n_runs=10, seed=3)
    assert probs["T00"]["champion"] == 1.0


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
