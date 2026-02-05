import logging

import numpy as np
import pandas as pd

from flat_prior_tests.priors.ml_prior_principled import MLPriorConfig, PrincipledMLPrior


def test_ml_prior_beta_params():
    # Minimal synthetic events to exercise feature pipeline.
    data = pd.DataFrame(
        {
            "ts_event": [1, 2, 3, 4],
            "ts_recv": [1, 2, 3, 4],
            "instrument_id": [1, 1, 1, 1],
            "action": ["A", "A", "F", "F"],
            "side": ["B", "S", "B", "S"],
            "price": [100.0, 101.0, 101.0, 100.5],
            "size": [10, 12, 5, 5],
            "order_id": [10, 11, 10, 11],
        }
    )

    cfg = MLPriorConfig(model_type="ensemble_logistic", n_models=4, lookback_events=10)
    prior = PrincipledMLPrior(cfg=cfg, logger=logging.getLogger(__name__))
    alpha, beta, diag = prior.predict_beta(data)

    assert alpha > 0
    assert beta > 0
    assert 0 < diag["mu"] < 1

