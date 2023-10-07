from collections import Counter

import jax.numpy as jnp
import numpy as np
import pytest

from dynagroup.hmm_posterior import compute_closed_form_M_step_on_posterior_summaries
from dynagroup.io import ensure_dir
from dynagroup.model2a.basketball.model import Model_Type, get_basketball_model
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    fit_rARHMM_to_bottom_half_of_model,
    make_data_free_preinitialization_of_ETP_JAX,
    make_data_free_preinitialization_of_IP_JAX,
    make_kmeans_preinitialization_of_CSP_JAX,
)
from dynagroup.params import Dims, EntityTransitionParameters_MetaSwitch_JAX
from dynagroup.vi.E_step import compute_log_entity_emissions_JAX, run_VEZ_step_JAX
from dynagroup.vi.M_step_and_ELBO import (
    run_M_step_for_CSP_in_closed_form__Gaussian_case,
)


###
# Fixtures
###


@pytest.fixture
def continuous_states():
    # Continuous states (normalized court lcoations) from CLE Starters dataset.
    # See `EXPORTED_DATA_README.md`
    return np.load("tests/artifacts/basketball/continuous_states.npy")


@pytest.fixture
def example_end_times():
    # Example end times from CLE Starters dataset.
    # See `EXPORTED_DATA_README.md`
    return np.load("tests/artifacts/basketball/example_end_times.npy")


@pytest.fixture
def DIMS(continuous_states):
    J, D = np.shape(continuous_states)[1:3]
    D_t = 2
    N = 0
    M_s, M_e = 0, 0  # for now!
    K = 10
    L = 5
    return Dims(J, K, L, D, D_t, N, M_s, M_e)


@pytest.fixture
def IP_JAX(DIMS):
    return make_data_free_preinitialization_of_IP_JAX(DIMS)


@pytest.fixture
def ETP_JAX(DIMS):
    return make_data_free_preinitialization_of_ETP_JAX(DIMS, method_for_Psis="rnorm", seed=0)  # Psis is (J, L, K, D_t)


@pytest.fixture
def CSP_JAX_and_kms(DIMS, continuous_states, example_end_times):
    ### Specify other arguments
    preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.DERIVATIVE

    ### Specify Output directory
    output_dir = "tests/outputs/"
    ensure_dir(output_dir)

    CSP_JAX, kms = make_kmeans_preinitialization_of_CSP_JAX(
        DIMS,
        continuous_states,
        preinitialization_strategy_for_CSP,
        example_end_times,
        save_dir=output_dir,
        plotbose=False,
    )
    return CSP_JAX, kms


@pytest.fixture
def model_basketball():
    return get_basketball_model(Model_Type.Linear_Recurrence)


###
# Helper functions
###
def _l1_distance_of_counts_from_uniform(counts: Counter) -> float:
    K = len(counts)  # number of categories
    N = np.sum(list(counts.values()))  # number of observations
    p_empirical = [x / N for x in counts.values()]
    p_uniform = [1 / K for k in range(K)]
    return np.max([np.abs(x - y) for (x, y) in zip(p_empirical, p_uniform)])


###
# Test functions
###


def test_that_gaussian_initialization_diversely_assigns_entity_states_to_basketball_data(
    continuous_states,
    example_end_times,
    DIMS,
    IP_JAX,
    ETP_JAX,
    CSP_JAX_and_kms,
    model_basketball,
):
    """
    We check that the soft assignments of observations to entity-level states is sufficiently diverse
    after initialization by the bottom-level rARHMM.
    """

    ### TODO: This function currently takes about 25 secs to run. Consider making it faster by just using a piece of
    ### the continuous_states.  Maybe could just run on a few players, or for fewer timesteps.

    ### Up front material
    CSP_JAX, _ = CSP_JAX_and_kms

    ### Function to be tested
    results_bottom = fit_rARHMM_to_bottom_half_of_model(
        continuous_states,
        example_end_times,
        CSP_JAX,
        ETP_JAX,
        IP_JAX,
        model_basketball,
        num_EM_iterations=5,
        use_continuous_states=None,
        params_frozen=None,
        verbose=True,
    )

    ### Now test that clusters are being diversely used even after initialization of bottom-level rARHMM
    THRESHOLD_l1_distance_of_counts_from_uniform = 0.20
    for j in range(DIMS.J):
        # player index; j=0 corresponds to Lebron James.

        labels_post_bottom_level_ARHMM = np.array(np.argmax(results_bottom.EZ_summaries.expected_regimes[:, j], 1))
        counts_labels_post_bottom_level_ARHMM = Counter(labels_post_bottom_level_ARHMM)
        print(
            f"Counts of entity-level state assigments after bottom-level rARHMM initialization for player {j}: {counts_labels_post_bottom_level_ARHMM}"
        )
        assert (
            _l1_distance_of_counts_from_uniform(counts_labels_post_bottom_level_ARHMM)
            <= THRESHOLD_l1_distance_of_counts_from_uniform
        )


@pytest.mark.skip(
    reason=f"This can be used to do a more detailed check than what is done by the more macroscopic test called "
    f"`test_that_gaussian_initialization_diversely_assigns_entity_states_to_basketball_data`.  However "
    f"if the macroscopic test passes, there is no need to additionally run this function, "
    f"since both have relatively long runtime."
)
def test_that_gaussian_initialization_diversely_assigns_entity_states_to_basketball_data__check_on_initial_steps(
    continuous_states,
    example_end_times,
    DIMS,
    IP_JAX,
    ETP_JAX,
    CSP_JAX_and_kms,
    model_basketball,
):
    """
    A more detailed analogue of `test_that_gaussian_initialization_diversely_assigns_entity_states_to_basketball_data`.
    We check that the assignments of observations to entity-level states is sufficiently diverse
    after initialization by the bottom-level rARHMM, but here we perform a check after each substep
    of the process. In particular, we run through the first few functions in the
    `fit_rARHMM_to_bottom_half_of_model` function from gaussian.initialize, and make a check after each function call.
    """

    ### Up front material
    use_continuous_states = None
    CSP_JAX, kms = CSP_JAX_and_kms

    ### Now partially reproduce the code body from `fit_rARHMM_to_bottom_half_of_model` that wasn't in the CSP_JAX fixture
    log_entity_emissions = compute_log_entity_emissions_JAX(
        CSP_JAX,
        IP_JAX,
        continuous_states,
        model_basketball,
        example_end_times,
    )

    T = len(continuous_states)
    VES_expected_regimes__uniform = np.ones((T, DIMS.L)) / DIMS.L
    EZ_summaries = run_VEZ_step_JAX(
        CSP_JAX,
        ETP_JAX,
        IP_JAX,
        continuous_states,
        VES_expected_regimes__uniform,
        model_basketball,
        example_end_times,
    )

    tpms = compute_closed_form_M_step_on_posterior_summaries(
        EZ_summaries,
        use_continuous_states,
        example_end_times,
    )
    Ps_new = jnp.tile(jnp.log(tpms[:, None, :, :]), (1, DIMS.L, 1, 1))
    ETP_JAX = EntityTransitionParameters_MetaSwitch_JAX(ETP_JAX.Psis, ETP_JAX.Omegas, Ps_new)
    CSP_JAX = run_M_step_for_CSP_in_closed_form__Gaussian_case(
        EZ_summaries.expected_regimes, continuous_states, example_end_times
    )
    log_entity_emissions__post_M_step = compute_log_entity_emissions_JAX(
        CSP_JAX,
        IP_JAX,
        continuous_states,
        model_basketball,
        example_end_times,
    )

    ### Now test that clusters are being diversely used even after each substep of initialization
    THRESHOLD_l1_distance_of_counts_from_uniform = 0.20
    for j in range(DIMS.J):
        # player index; j=0 corresponds to Lebron James.

        print(f"-- Now checking cluster assigments during initialization for entity {j}")

        ### Check that assignments to clusters is sufficiently diverse  according to various
        ### subcomponents of the initialization process ...

        ### ....according to k-means pre-initialization
        labels_pre_init = kms[j].labels_
        counts_pre_init = Counter(labels_pre_init)
        print(f"Counts init: {counts_pre_init}")
        assert _l1_distance_of_counts_from_uniform(counts_pre_init) <= THRESHOLD_l1_distance_of_counts_from_uniform

        ### ....according to entity-level state indicators which maximize emissions probability
        labels_emiss = np.array(np.argmax(log_entity_emissions[1:, j, :], 1))
        counts_emiss = Counter(labels_emiss)
        print(f"Counts post-emission: {counts_emiss}")
        assert _l1_distance_of_counts_from_uniform(counts_emiss) <= THRESHOLD_l1_distance_of_counts_from_uniform

        ### ....according to entity-level state indicators which maximize E-step probabilities
        labels_post_E_step = np.array(np.argmax(EZ_summaries.expected_regimes[:, j], 1))
        counts_labels_post_E_step = Counter(labels_post_E_step)
        print(f"Counts post E-step: {counts_labels_post_E_step}")
        assert (
            _l1_distance_of_counts_from_uniform(counts_labels_post_E_step)
            <= THRESHOLD_l1_distance_of_counts_from_uniform
        )

        ### ... according to M-step using the E-step probabilities
        labels_post_M_step = np.array(np.argmax(log_entity_emissions__post_M_step[1:, j, :], 1))
        counts_post_M_step = Counter(labels_post_M_step)
        print(f"Counts post-M-step: {counts_post_M_step}")
        assert _l1_distance_of_counts_from_uniform(counts_post_M_step) <= THRESHOLD_l1_distance_of_counts_from_uniform

        # TODO: In addition to checking that we don't have the "disappearing cluster" problem, check that assignments are internally
        # consistent.  That is, if we compute the discrete derivatives from previous continuous_state to the one assigned to cluster k,
        # we get things that look reasonable.   Find a way to do this quantitatively, e.g. perhaps using an anomaly score after
        # applying a simple model to the cluster-specific derivatives.
