from typing import Optional, Tuple

import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.initialize import InitializationResults
from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.model import Model
from dynagroup.params import AllParameters_JAX
from dynagroup.types import JaxNumpyArray2D, JaxNumpyArray3D, NumpyArray1D, NumpyArray2D
from dynagroup.vi.E_step import run_VES_step_JAX, run_VEZ_step_JAX
from dynagroup.vi.M_step_and_ELBO import (
    M_Step_Toggles,
    compute_elbo_decomposed,
    run_M_step_for_CSP,
    run_M_step_for_ETP,
    run_M_step_for_IP,
    run_M_step_for_STP,
)
from dynagroup.vi.prior import SystemTransitionPrior_JAX


def run_CAVI_with_JAX(
    continuous_states: JaxNumpyArray3D,
    n_iterations: int,
    initialization_results: InitializationResults,
    model: Model,
    M_step_toggles: Optional[M_Step_Toggles] = None,
    num_M_step_iters: int = 50,
    system_transition_prior: Optional[SystemTransitionPrior_JAX] = None,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    true_system_regimes: Optional[NumpyArray1D] = None,
    true_entity_regimes: Optional[NumpyArray2D] = None,
    verbose: bool = True,
) -> Tuple[HMM_Posterior_Summary_JAX, HMM_Posterior_Summaries_JAX, AllParameters_JAX]:
    """
    Arguments:
        continuous_states: jnp.array with shape (T, J, D)
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.
        M_step_toggles: Describes what kind of M-step should be done (gradient-descent, closed-form, or None)
            for each subclass of parameters (STP, ETP, CSP, IP).

            As of 4/20/23, supported values are:
                STP: Closed-form, gradient decent, or none
                ETP: Gradient decent, or none
                CSP: Closed-form, gradient decent, or none  (but gradient descent doesn't work very well)
                IP: Closed-form or none
        true_system_regimes: Array with shape (T,)
            Each entry is in {1,...,L}
        true_entity_regimes: has shape (T, J)
            Each entry is in {1,...,K}

    Returns:
        VES_Summary, VEZ_Summaries, all parameters.

    Notation:
        T: number of timesteps
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimension of continuous states
    """

    ###
    # SET-UP
    ###

    IR = initialization_results
    all_params, VES_summary, VEZ_summaries = IR.params, IR.ES_summary, IR.EZ_summaries
    J = np.shape(continuous_states)[1]

    # TODO:  I need to have a way to do a DUMB (default/non-data-informed) init for both VEZ and VES summaries
    # so that we can get ELBO baselines BEFORE the smart-initialization.... Maybe make VEZ, VES uniform? And
    # use the data-free inits for everything else?
    #
    if verbose:
        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            system_covariates,
        )
        print(
            f"After (possibly smart) initialization, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
        )

    ###
    # CAVI
    ###

    for i in range(n_iterations):
        print(f"\n ---- Now running iteration {i+1} ----")

        VES_summary = run_VES_step_JAX(
            all_params.STP,
            all_params.ETP,
            all_params.IP,
            continuous_states,
            VEZ_summaries,
            model,
            system_covariates,
        )

        if verbose:
            print(
                f"\nVES step's log normalizer for entity regimes when we use uniform inits for q(Z): {VES_summary.log_normalizer:.02f}"
            )
            if true_system_regimes is not None:
                most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
                pct_correct_system = compute_regime_labeling_accuracy(
                    most_likely_system_regimes, true_system_regimes
                )
                print(
                    f"Percent correct classifications for system segmentations {pct_correct_system:.02f}"
                )

        VEZ_summaries = run_VEZ_step_JAX(
            all_params.CSP,
            all_params.ETP,
            all_params.IP,
            continuous_states,
            VES_summary.expected_regimes,
            model,
        )

        if verbose:
            print(
                f"\nVEZ step's log normalizer by entities for continuous state emissions when we use VES inits for q(S): {VEZ_summaries.log_normalizers}"
            )
            if true_entity_regimes is not None:
                pct_corrects_entities = np.empty(J)
                for j in range(J):
                    most_likely_system_regimes = np.argmax(
                        VEZ_summaries.expected_regimes[:, j, :], axis=1
                    )
                    pct_corrects_entities[j] = compute_regime_labeling_accuracy(
                        most_likely_system_regimes, true_entity_regimes[:, j]
                    )
                print(
                    f"Percent correct classifications for entity-level segmentations {pct_corrects_entities}"
                )

            elbo_decomposed = compute_elbo_decomposed(
                all_params,
                VES_summary,
                VEZ_summaries,
                system_transition_prior,
                continuous_states,
                model,
                system_covariates,
            )
            print(
                f"After E-step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        # TODO: I probably don't really need separate functions of the form run_M_step_for_<xxxx>.  Make this a single wrapper that in
        # turn calls the appropriate functions for closed-form or gradient descent inference.

        ###
        # M-step (ETP)
        ###

        all_params = run_M_step_for_ETP(
            all_params,
            M_step_toggles.ETP,
            VES_summary,
            VEZ_summaries,
            continuous_states,
            i,
            num_M_step_iters,
            model,
            verbose,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            system_covariates,
        )
        if verbose:
            print(
                f"After ETP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        ###
        # M-step (STP)
        ###

        all_params = run_M_step_for_STP(
            all_params,
            M_step_toggles.STP,
            VES_summary,
            system_transition_prior,
            i,
            num_M_step_iters,
            model,
            system_covariates,
            verbose,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            system_covariates,
        )
        if verbose:
            print(
                f"After STP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        ###
        # M-step (CSP)
        ###

        all_params = run_M_step_for_CSP(
            all_params,
            M_step_toggles.CSP,
            VEZ_summaries,
            continuous_states,
            i,
            num_M_step_iters,
            model,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            system_covariates,
        )
        if verbose:
            print(
                f"After CSP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        ###
        # M-step (IP)
        ###

        all_params = run_M_step_for_IP(
            all_params,
            M_step_toggles.IP,
            VES_summary,
            VEZ_summaries,
            continuous_states,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            system_covariates,
        )
        if verbose:
            print(
                f"After IP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

    return VES_summary, VEZ_summaries, all_params
