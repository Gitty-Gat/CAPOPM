from flat_prior_tests.priors.hybrid_mixture_prior import MixturePriorState


def test_mixture_weight_normalization():
    state = MixturePriorState(
        alpha_ml=2.0,
        beta_ml=3.0,
        alpha_flat=1.0,
        beta_flat=1.0,
        weight=1.5,  # intentionally invalid
    )
    normalized = state.normalize_weight()
    assert 0.0 <= normalized.weight <= 1.0
