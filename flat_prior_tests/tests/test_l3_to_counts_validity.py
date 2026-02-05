import pandas as pd

from flat_prior_tests.simulation.map_l3_to_counts import map_l3_to_counts


def test_l3_to_counts_validity():
    events = pd.DataFrame(
        {
            "ts_event": [1, 2, 3],
            "ts_recv": [1, 2, 3],
            "instrument_id": [1, 1, 1],
            "action": ["A", "A", "F"],
            "side": ["B", "S", "B"],
            "price": [100.0, 101.0, 101.5],
            "size": [5, 5, 5],
            "order_id": [1, 2, 1],
        }
    )

    y, n, diag = map_l3_to_counts(events)
    assert n == 1
    assert y in (0, 1)
    assert len(diag["fills"]) == 1

