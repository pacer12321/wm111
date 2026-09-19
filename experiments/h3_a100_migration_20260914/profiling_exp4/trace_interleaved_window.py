from __future__ import annotations


GLOBAL_ROWS = 81216
WORLD_SIZE = 2
LOCAL_ROWS = GLOBAL_ROWS // WORLD_SIZE
TARGET_START = 6159 + 37296
TOKENS_PER_FRAME = 24 * 42


def physical_index(logical_index: int) -> int:
    """Rank-concatenated sequence index after even/odd sharding + Ulysses."""
    owner = logical_index % WORLD_SIZE
    local_index = logical_index // WORLD_SIZE
    return owner * LOCAL_ROWS + local_index


def logical_index(physical_index_: int) -> int:
    owner = physical_index_ // LOCAL_ROWS
    local_index = physical_index_ % LOCAL_ROWS
    return local_index * WORLD_SIZE + owner


def main() -> None:
    # The first two spatial neighbours in target frame 1. Frame 1 is an
    # interior OpenVDN frame and therefore uses the local frame-window route.
    left = TARGET_START + TOKENS_PER_FRAME
    right = left + 1

    assert left % 2 == 1 and right % 2 == 0
    assert left // 2 < LOCAL_ROWS and right // 2 < LOCAL_ROWS

    left_physical = physical_index(left)
    right_physical = physical_index(right)
    assert logical_index(left_physical) == left
    assert logical_index(right_physical) == right

    # Ulysses pre_attention returns GLOBAL_ROWS sequence positions on every
    # attention rank (with only heads sharded), so both mapped positions exist.
    assert 0 <= left_physical < GLOBAL_ROWS
    assert 0 <= right_physical < GLOBAL_ROWS

    # Demonstrate why the current implicit frame indices would be wrong after
    # interleaving: treating logical indices as physical indices selects other
    # original tokens.
    wrong_left_logical = logical_index(left)
    wrong_right_logical = logical_index(right)
    assert (wrong_left_logical, wrong_right_logical) != (left, right)

    print(f"logical neighbours: {left}, {right}")
    print(f"pre-Ulysses owners: rank{left % 2}, rank{right % 2}")
    print(f"local indices: {left // 2}, {right // 2}")
    print(f"post-Ulysses physical indices: {left_physical}, {right_physical}")
    print("post-Ulysses visibility: both positions are present on every attention rank")
    print(
        "without logical->physical mapping, current window indices would select "
        f"logical tokens {wrong_left_logical}, {wrong_right_logical} instead"
    )


if __name__ == "__main__":
    main()
