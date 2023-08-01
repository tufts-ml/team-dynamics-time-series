from dataclasses import dataclass
from itertools import groupby
from typing import Optional

import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.model import Model
from dynagroup.params import (
    AllParameters_JAX,
    ContinuousStateParameters_JAX,
    EmissionsParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    InitializationParameters_JAX,
    SystemTransitionParameters_JAX,
)
from dynagroup.types import (
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
)
from dynagroup.vi.M_step_and_ELBO import ELBO_Decomposed, compute_elbo_decomposed
from dynagroup.vi.prior import SystemTransitionPrior_JAX


###
# STRUCTS
###


@dataclass
class InitializationResults:
    params: AllParameters_JAX
    ES_summary: HMM_Posterior_Summary_JAX
    EZ_summaries: HMM_Posterior_Summaries_JAX
    record_of_most_likely_system_states: NumpyArray2D  # Txnum_EM_iterations
    record_of_most_likely_entity_states: NumpyArray3D  # TxJx num_EM_iterations


@dataclass
class ResultsFromBottomHalfInit:
    """
    Attributes:
        record_of_most_likely_states:  Has shape (T,J,num_EM_iterations).
            Note that this is NOT most likely in the Viterbi sense,
            it's just the argmax from the expected unary marginals.
    """

    CSP: ContinuousStateParameters_JAX
    EZ_summaries: HMM_Posterior_Summaries_JAX
    record_of_most_likely_states: NumpyArray3D  # TxJx num_EM_iterations


@dataclass
class ResultsFromTopHalfInit:
    """
    Attributes:
        record_of_most_likely_states:  Has shape (T, num_EM_iterations).
            Note that this is NOT most likely in the Viterbi sense,
            it's just the argmax from the expected unary marginals.
    """

    STP: SystemTransitionParameters_JAX
    ETP: EntityTransitionParameters_MetaSwitch_JAX
    ES_summary: HMM_Posterior_Summary_JAX
    record_of_most_likely_states: NumpyArray2D  # Txnum_EM_iterations


@dataclass
class RawInitializationResults:
    """
    Compared to `InitializationResults`, this representation is closer to how the initialization was constructed:
    there's info from a "bottom-level" AR-HMM and from a "top-level" AR-HMM.

    These results are also useful for inspecting the quality of the initialization
    (e.g. via `top.record_of_most_likely_states` or `bottom.record_of_most_likely_states`)
    with respect to known truth.
    """

    bottom: ResultsFromBottomHalfInit
    top: ResultsFromTopHalfInit
    IP: InitializationParameters_JAX
    EP: EmissionsParameters_JAX


def initialization_results_from_raw_initialization_results(
    raw_initialization_results: RawInitializationResults,
    params_frozen: Optional[AllParameters_JAX] = None,
):
    RI = raw_initialization_results
    if params_frozen:
        params = params_frozen
    else:
        params = AllParameters_JAX(RI.top.STP, RI.top.ETP, RI.bottom.CSP, RI.EP, RI.IP)
    return InitializationResults(
        params,
        RI.top.ES_summary,
        RI.bottom.EZ_summaries,
        RI.top.record_of_most_likely_states,
        RI.bottom.record_of_most_likely_states,
    )


###
# DIAGNOSTICS
###


def inspect_entity_level_segmentations_over_EM_iterations(
    record_of_most_likely_states: NumpyArray3D,
    zs_true: NumpyArray2D,
) -> None:
    """
    Arguments:
        record_of_most_likely_states:  An attribute from the ResultsFromBottomHalfInit class.
            Has shape (T,J,num_EM_iterations).  Note that this is NOT most likely in the Viterbi sense,
            it's just the argmax from the expected unary marginals.
        zs_true: Has shape (T, J).  Each entry is in {1,...,K}.  Can be grabbed from the Sample class.
    """
    _, J, num_EM_iterations = np.shape(record_of_most_likely_states)

    print(
        "\n---Now inspecting the learning (during initialization) of the entity-level segmentations.---"
    )
    for j in range(J):
        print(f"\n\nNow investigating entity {j}....")
        for i in range(num_EM_iterations):
            most_likely_states = record_of_most_likely_states[:, j, i]
            count_dups_estimated = [
                sum(1 for _ in group) for _, group in groupby(most_likely_states)
            ]
            count_dups_true = [sum(1 for _ in group) for _, group in groupby(zs_true[:, j])]
            print(
                f"For entity {j}, after EM it {i+1}, number of consecutive duplications for estimated: {count_dups_estimated}. For true: {count_dups_true}"
            )


def inspect_system_level_segmentations_over_EM_iterations(
    record_of_most_likely_states: NumpyArray2D,
    s_true: NumpyArray1D,
) -> None:
    """
    Arguments:
        record_of_most_likely_states:  An attribute from the ResultsFromTopHalfInit class.
            Has shape (T,num_EM_iterations).  Note that this is NOT most likely in the Viterbi sense,
            it's just the argmax from the expected unary marginals.
        s_true: Has shape (T).  Each entry is in {1,...,L}.  Can be grabbed from the Sample class.
    """
    _, num_EM_iterations = np.shape(record_of_most_likely_states)

    print(
        "\n---Now inspecting the learning (during initialization) of the system-level segmentations.---"
    )
    for i in range(num_EM_iterations):
        most_likely_states = record_of_most_likely_states[:, i]
        count_dups_estimated = [sum(1 for _ in group) for _, group in groupby(most_likely_states)]
        count_dups_true = [sum(1 for _ in group) for _, group in groupby(s_true)]
        print(
            f"After EM it {i}, number of consecutive duplications for estimated: {count_dups_estimated}. For true: {count_dups_true}"
        )


def compute_elbo_from_initialization_results(
    initialization_results: InitializationResults,
    system_transition_prior: SystemTransitionPrior_JAX,
    continuous_states: JaxNumpyArray3D,
    model: Model,
    event_end_times: Optional[NumpyArray1D],
    system_covariates: Optional[JaxNumpyArray2D],
) -> ELBO_Decomposed:
    if event_end_times is None:
        T = len(continuous_states)
        event_end_times = np.array([-1, T])

    elbo_decomposed = compute_elbo_decomposed(
        initialization_results.params,
        initialization_results.ES_summary,
        initialization_results.EZ_summaries,
        system_transition_prior,
        continuous_states,
        model,
        event_end_times,
        system_covariates,
    )
    return elbo_decomposed.elbo
