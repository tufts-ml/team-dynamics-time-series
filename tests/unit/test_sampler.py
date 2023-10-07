import numpy as np
import numpy.random as npr
import pytest

from dynagroup.model2a.basketball.model import Model_Type, get_basketball_model
from dynagroup.model2a.gaussian.initialize import (
    make_data_free_preinitialization_of_EP_JAX,
    make_data_free_preinitialization_of_IP_JAX,
    make_data_free_preinitialization_of_STP_JAX,
    make_tpm_only_preinitialization_of_ETP_JAX,
)
from dynagroup.params import (
    AllParameters_JAX,
    ContinuousStateParameters_Gaussian_JAX,
    Dims,
)
from dynagroup.sampler import (
    get_multiple_samples_of_team_dynamics,
    sample_team_dynamics,
)


@pytest.fixture
def DIMS() -> Dims:
    K = 10
    L = 5
    J = 10
    D = 2
    D_t = 2
    N = 0
    M_s, M_e = 0, 0  # for now!
    return Dims(J, K, L, D, D_t, N, M_s, M_e)


def make_CSP_JAX___with_identity_state_matrix(
    DIMS: Dims, shared_variance=1.0
) -> ContinuousStateParameters_Gaussian_JAX:
    As = np.tile(np.eye(DIMS.D)[None, None, :, :], (DIMS.J, DIMS.K, 1, 1))
    bs = npr.randn(DIMS.J, DIMS.K, DIMS.D)
    Qs = np.tile(shared_variance * np.eye(DIMS.D)[None, None, :, :], (DIMS.J, DIMS.K, 1, 1))
    return ContinuousStateParameters_Gaussian_JAX(As, bs, Qs)


@pytest.fixture
def params(DIMS) -> AllParameters_JAX:
    # TODO: Support fixed or random draws from prior.
    ETP_JAX = make_tpm_only_preinitialization_of_ETP_JAX(DIMS, fixed_self_transition_prob=0.90)
    # TODO: Support fixed or random draws from prior.
    IP_JAX = make_data_free_preinitialization_of_IP_JAX(DIMS, shared_variance=1.0)
    # EP_JAX is a placeholder; not used for Figure 8.
    EP_JAX = make_data_free_preinitialization_of_EP_JAX(DIMS)

    STP_JAX = make_data_free_preinitialization_of_STP_JAX(
        DIMS, method_for_Upsilon="rnorm", fixed_self_transition_prob=0.90, seed=0
    )

    CSP_JAX = make_CSP_JAX___with_identity_state_matrix(DIMS, shared_variance=1.0)
    return AllParameters_JAX(STP_JAX, ETP_JAX, CSP_JAX, EP_JAX, IP_JAX)


@pytest.fixture
def params__for_nearly_deterministic_stepping(DIMS) -> AllParameters_JAX:
    # TODO: Support fixed or random draws from prior.
    ETP_JAX = make_tpm_only_preinitialization_of_ETP_JAX(DIMS, fixed_self_transition_prob=(1.0 - 1e-10))
    # TODO: Support fixed or random draws from prior.
    IP_JAX = make_data_free_preinitialization_of_IP_JAX(DIMS, shared_variance=1e-6)
    # EP_JAX is a placeholder; not used for Figure 8.
    EP_JAX = make_data_free_preinitialization_of_EP_JAX(DIMS)

    STP_JAX = make_data_free_preinitialization_of_STP_JAX(
        DIMS, method_for_Upsilon="rnorm", fixed_self_transition_prob=(1.0 - 1e-10), seed=0
    )

    CSP_JAX = make_CSP_JAX___with_identity_state_matrix(DIMS, shared_variance=1e-6)
    return AllParameters_JAX(STP_JAX, ETP_JAX, CSP_JAX, EP_JAX, IP_JAX)


@pytest.fixture
def model_with_linear_recurrence():
    return get_basketball_model(Model_Type.Linear_Recurrence)


@pytest.fixture
def model_with_no_recurrence():
    return get_basketball_model(Model_Type.No_Recurrence)


def test_that_sampling_can_be_controlled_with_random_seed(params, model_with_linear_recurrence):
    T = 12
    num_samples = 3

    samples_with_seed_0 = get_multiple_samples_of_team_dynamics(
        num_samples, params, T, model_with_linear_recurrence, seed=0
    )
    samples_with_seed_0_again = get_multiple_samples_of_team_dynamics(
        num_samples, params, T, model_with_linear_recurrence, seed=0
    )
    samples_with_seed_1 = get_multiple_samples_of_team_dynamics(
        num_samples, params, T, model_with_linear_recurrence, seed=1
    )

    for sample_idx in range(num_samples):
        samples_match_when_same_seed_is_used = (
            samples_with_seed_0[sample_idx].xs == samples_with_seed_0_again[sample_idx].xs
        ).all()
        assert samples_match_when_same_seed_is_used

        samples_differ_when_different_seeds_are_used = (
            samples_with_seed_0[sample_idx].xs != samples_with_seed_1[sample_idx].xs
        ).all()
        assert samples_differ_when_different_seeds_are_used


def test_that_multiple_samples_are_diverse(params, model_with_linear_recurrence):
    T = 12
    num_samples = 2

    samples = get_multiple_samples_of_team_dynamics(
        num_samples,
        params,
        T,
        model_with_linear_recurrence,
    )

    sample_idx, another_sample_idx = 0, 1

    # RK: this assertion (effectively) IMPLIES that higher-level items in the generative model
    # are different from sample to sample, like the z-probabilities.
    # I confirmed this fact interactively.  Consider adding
    # an explicit test as well.
    multiple_samples_differ_in_their_generated_xs = (samples[sample_idx].xs != samples[another_sample_idx].xs).all()
    assert multiple_samples_differ_in_their_generated_xs


def test_that_a_generated_sample_looks_as_expected_under_little_noise_and_no_recurrence(
    params__for_nearly_deterministic_stepping,
    model_with_no_recurrence,
    DIMS,
):
    T = 12

    sample = sample_team_dynamics(
        params__for_nearly_deterministic_stepping,
        T,
        model_with_no_recurrence,
        fixed_init_system_regime=0,
        fixed_init_entity_regimes=[0] * DIMS.J,
        seed=0,
    )

    ### by fixing the initial system, entity regimes, ablating recurrence, and making the matrices very sticky
    ### i should have guaranteed that the z's are uniformly 0 for all (T,J)
    expected_zs = np.zeros((T, DIMS.J))
    assert np.isclose(sample.zs, expected_zs).all()

    ### so given that, we can expect that the params of the 0th entity state are governing the dynamics
    uniform_entity_state = 0

    ### and since the continuous state dynamics are almost deterministic (with an identity transformation
    ### on the previous state, plus a step size b, plus almost no noise), we should basically expect the
    ### observations to just be equal to the step size (for a given entity j and state k) times the number
    ### of time steps.
    bs_active_for_each_entity = params__for_nearly_deterministic_stepping.CSP.bs[:, uniform_entity_state, :]

    for t in range(T):
        expected_xs = t * bs_active_for_each_entity
        assert np.isclose(sample.xs[t], expected_xs, atol=0.10).all()
