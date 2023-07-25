from dynagroup.model2a.basketball.data.baller2vec.positions import (
    get_opponent_player_names_ordered_by_position,
    make_position_group_to_player_names,
)


OPPONENT_NAMES_2_POSITIONS = {
    "Marvin Williams": "Power forward / small forward",
    "Kemba Walker": "Point guard",
    "PJ Hairston": "Small forward / shooting guard",
    "Cody Zeller": "Center / power forward",
    "Nicolas Batum": "Power forward / small forward",
}


def test_get_opponent_player_names_ordered_by_position():
    opponent_player_names_ordered_by_position_computed = (
        get_opponent_player_names_ordered_by_position(
            OPPONENT_NAMES_2_POSITIONS,
        )
    )
    opponent_player_names_ordered_by_position_expected = [
        "Marvin Williams",
        "Nicolas Batum",
        "Cody Zeller",
        "PJ Hairston",
        "Kemba Walker",
    ]
    assert (
        opponent_player_names_ordered_by_position_computed
        == opponent_player_names_ordered_by_position_expected
    )


def test_make_position_group_to_player_names():
    position_group_to_player_names_computed = make_position_group_to_player_names(
        OPPONENT_NAMES_2_POSITIONS
    )

    position_group_to_player_names_expected = {
        "forward": ["Marvin Williams", "Nicolas Batum"],
        "center": ["Cody Zeller"],
        "guard": ["Kemba Walker", "PJ Hairston"],
    }

    assert position_group_to_player_names_computed == position_group_to_player_names_expected
