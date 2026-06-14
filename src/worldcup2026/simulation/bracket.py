"""FIFA's fixed 2026 World Cup knockout bracket.

The 48-team format sends the 12 group winners, 12 runners-up and the 8 best
third-placed teams into a 32-team knockout. Unlike a generic seeded bracket,
the pairings are *pre-published*: every Round-of-32 match has fixed slots for a
specific group winner / runner-up, and the winners feed a fixed tree through to
the final (FIFA match numbers 73-104).

The only piece that depends on the group-stage outcome is which of the eight
third-placed slots each qualifying third-placed team takes. FIFA's Annex C
fixes that with a 495-row table; we don't reproduce the table verbatim, but we
*do* honour its hard constraints — each third-place slot only accepts teams
from a published set of groups, and a team never meets its own group — and pick
a deterministic assignment satisfying them. See ``assign_thirds``.

Slot encoding:
    ("W", "E")              -> winner of group E
    ("R", "C")              -> runner-up of group C
    ("3", frozenset("AB..")) -> a best third-placed team from one of those groups
"""

from __future__ import annotations

Slot = tuple[str, object]

# The 16 Round-of-32 matches, keyed by FIFA match number (73-88).
R32_MATCHES: dict[int, tuple[Slot, Slot]] = {
    73: (("R", "A"), ("R", "B")),
    74: (("W", "E"), ("3", frozenset("ABCDF"))),
    75: (("W", "F"), ("R", "C")),
    76: (("W", "C"), ("R", "F")),
    77: (("W", "I"), ("3", frozenset("CDFGH"))),
    78: (("R", "E"), ("R", "I")),
    79: (("W", "A"), ("3", frozenset("CEFHI"))),
    80: (("W", "L"), ("3", frozenset("EHIJK"))),
    81: (("W", "D"), ("3", frozenset("BEFIJ"))),
    82: (("W", "G"), ("3", frozenset("AEHIJ"))),
    83: (("R", "K"), ("R", "L")),
    84: (("W", "H"), ("R", "J")),
    85: (("W", "B"), ("3", frozenset("EFGIJ"))),
    86: (("W", "J"), ("R", "H")),
    87: (("W", "K"), ("3", frozenset("DEIJL"))),
    88: (("R", "D"), ("R", "G")),
}

# Round-of-32 winners feed this fixed tree (match -> the two feeding matches).
KNOCKOUT_TREE: dict[int, tuple[int, int]] = {
    89: (74, 77), 90: (73, 75), 91: (76, 78), 92: (79, 80),
    93: (83, 84), 94: (81, 82), 95: (86, 88), 96: (85, 87),  # round of 16
    97: (89, 90), 98: (93, 94), 99: (91, 92), 100: (95, 96),  # quarter-finals
    101: (97, 98), 102: (99, 100),                            # semi-finals
    104: (101, 102),                                          # final
}

# Match number where each third-place slot lives -> the groups it may draw from.
THIRD_SLOT_GROUPS: dict[int, frozenset[str]] = {
    match: slot2[1]
    for match, (_slot1, slot2) in R32_MATCHES.items()
    if slot2[0] == "3"
}

GROUP_LETTERS = tuple("ABCDEFGHIJKL")


def _leaf_order() -> list[int]:
    """R32 match numbers in left-to-right bracket-leaf order.

    Folding winners pairwise in this order reproduces ``KNOCKOUT_TREE`` exactly,
    so a flat fold loop yields the official bracket.
    """
    order: list[int] = []

    def walk(match: int) -> None:
        if match in R32_MATCHES:
            order.append(match)
        else:
            left, right = KNOCKOUT_TREE[match]
            walk(left)
            walk(right)

    walk(104)  # the final is the root
    return order


R32_LEAF_ORDER: tuple[int, ...] = tuple(_leaf_order())


def assign_thirds(qualified_groups: list[str]) -> dict[int, str]:
    """Assign the 8 qualifying third-placed groups to the 8 R32 third slots.

    `qualified_groups` is the set of group letters whose third-placed team made
    the cut. Returns ``{match_number: group_letter}``. The assignment honours
    FIFA's published per-slot group sets (``THIRD_SLOT_GROUPS``); among the
    valid assignments we take a deterministic one (slots filled in match-number
    order, candidate groups tried alphabetically). A valid assignment always
    exists for the 495 legal combinations.
    """
    if len(qualified_groups) != 8 or len(set(qualified_groups)) != 8:
        raise ValueError(f"need 8 distinct third-placed groups, got {qualified_groups}")
    qualified = set(qualified_groups)
    slots = sorted(THIRD_SLOT_GROUPS)  # match-number order
    assignment: dict[int, str] = {}
    used: set[str] = set()

    def backtrack(i: int) -> bool:
        if i == len(slots):
            return True
        match = slots[i]
        for g in sorted(THIRD_SLOT_GROUPS[match] & qualified):
            if g not in used:
                used.add(g)
                assignment[match] = g
                if backtrack(i + 1):
                    return True
                used.discard(g)
                del assignment[match]
        return False

    if not backtrack(0):
        raise ValueError(f"no valid third-place assignment for groups {sorted(qualified)}")
    return assignment


def _resolve(slot: Slot, winners, runners_up, thirds, third_assignment, match: int) -> str:
    kind = slot[0]
    if kind == "W":
        return winners[slot[1]]
    if kind == "R":
        return runners_up[slot[1]]
    # third-place slot: look up which group was routed to this match
    return thirds[third_assignment[match]]


def bracket_order(
    winners: dict[str, str],
    runners_up: dict[str, str],
    thirds: dict[str, str],
    third_assignment: dict[int, str],
) -> list[str]:
    """Resolve the bracket to the 32 teams in leaf order.

    `winners`/`runners_up` map every group letter to a team; `thirds` maps each
    *qualifying* third-placed group letter to its team. The returned list, folded
    pairwise, plays out the official bracket.
    """
    order: list[str] = []
    for match in R32_LEAF_ORDER:
        slot1, slot2 = R32_MATCHES[match]
        order.append(_resolve(slot1, winners, runners_up, thirds, third_assignment, match))
        order.append(_resolve(slot2, winners, runners_up, thirds, third_assignment, match))
    return order
