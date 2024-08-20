import warnings
from typing import Optional, Tuple, Union

import numpy as np

from dynagroup.examples import example_end_times_are_proper
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.initialize import InitializationResults
from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.model import Model
from dynagroup.params import AllParameters_JAX, dims_from_params
from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    NumpyArray1D,
    NumpyArray2D,
)
from dynagroup.vi.E_step_2 import run_VES_step_JAX, run_VEZ_step_JAX
from dynagroup.vi.M_step_and_ELBO import (
    ELBO_Decomposed,
    M_Step_Toggle_Value,
    M_Step_Toggles,
    compute_elbo_decomposed,
    run_M_step_for_CSP,
    run_M_step_for_ETP,
    run_M_step_for_IP,
    run_M_step_for_STP,
)
from dynagroup.vi.prior import SystemTransitionPrior_JAX


def run_CAVI_with_JAX(
    continuous_states: Union[JaxNumpyArray2D, JaxNumpyArray3D],  #IS THIS FOR MULTIPLE SEQUENCES?? 
    n_iterations: int,
    initialization_results: InitializationResults,
    model: Model,
    example_end_times: Optional[JaxNumpyArray1D] = None,
    M_step_toggles: Optional[M_Step_Toggles] = None,
    num_M_step_iters: int = 50,
    system_transition_prior: Optional[SystemTransitionPrior_JAX] = None,
    system_covariates: Optional[JaxNumpyArray2D] = None,
    use_continuous_states: Optional[NumpyArray2D] = None,
    true_system_regimes: Optional[NumpyArray1D] = None,
    true_entity_regimes: Optional[NumpyArray2D] = None,
    verbose: bool = True,
) -> Tuple[HMM_Posterior_Summary_JAX, HMM_Posterior_Summaries_JAX, AllParameters_JAX, ELBO_Decomposed]:
    """
    Arguments:
        continuous_states: jnp.array with shape (T,J) or (T, J, D)
            If (T,J), we assume this means (T,J,D) where D=1, and convert it to have 3 array dims.
        transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix: transform R^D -> R^D
            of the continuous state vector before pre-multiplying by the the recurrence matrix.
        example_end_times: optional, has shape (E+1,)
            An `example` (or event) takes an ordinary sampled group time series of shape (T,J,:) and interprets it
            as (T_grand,J,:), where T_grand is the sum of the number of timesteps across i.i.d "examples".
            An example might induce a largetime gap between timesteps, and a discontinuity in the continuous states x.

            If there are E examples, then along with the observations, we store
                end_times=[-1, t_1, …, t_E], where t_e is the timestep at which the e-th example ended.
            So to get the timesteps for the e-th example, you can index from 1,…,T_grand by doing
                    [end_times[e-1]+1 : end_times[e]].

        M_step_toggles: Describes what kind of M-step should be done (gradient-descent, closed-form, or None)
            for each subclass of parameters (STP, ETP, CSP, IP).

            As of 4/20/23, supported values are:
                STP: Closed-form, gradient decent, or none
                ETP: Gradient decent, or none
                CSP: Closed-form, gradient decent, or none  (but gradient descent doesn't work very well)
                IP: Closed-form or none
        use_continuous_states: If None, we assume all states should be utilized in inference.
            Otherwise, this is a (T,J) boolean vector such that the (t,j)-th element is True if
            continuous_states[t,j] should be utilized in inference and False otherwise.
            In particular, the M-step for the entity-level parameters and the VES step have no insight into this info.
            Note that because this functionality was added just prior on 5/12/23, just prior to the NeurIPS deadline,
            the VEZ step still has access to this info. But the mask prevents the VEZ step from feeding masked
            data into the M step and the VES step. The only glitch is that the ELBO ignores the mask,
            and is still always computed for the full data set. But the ELBO is not used to determine inference steps
            -- it's just computed post-hoc for informational purposes.
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
        E: number of examples
    """

    ###
    # SET-UP
    ###

    IR = initialization_results
    all_params, VES_summary, VEZ_summaries = IR.params, IR.ES_summary, IR.EZ_summaries
    DIMS = dims_from_params(all_params)
    T = np.shape(continuous_states)[0]

    if continuous_states.ndim == 2:
        print("Continuous states has only two array dimensions; now adding a third array dimension with 1 element.")
        continuous_states = continuous_states[:, :, None]

    if use_continuous_states is None:
        use_continuous_states = np.full((T, DIMS.J), True)
    else:
        # TODO: Raise error if use_continuous_states has False followed by True for any entity j;
        # in the current implementation, inference will not be done correctly, because the VEZ step will not correctly
        # remove the missing data -- so the inference after the missing data will be artificially good.
        warnings.warn(
            f"Selecting only some continuous states for usage correctly alters inference -- the M-step and VES "
            f"steps are changes so as to remove the influence of unused states.  However, the "
            f"ELBO is still computed on the full dataset. This is because of the current implementation: under "
            f"the hood, we compute the full VEZ step, and only ablate post-hoc."
        )

    if False in use_continuous_states[0]:
        raise NotImplementedError(
            f"We currently assume the initial continuous state is used for all entities. "
            f"The implementation can be changed to handle this case, though: Update the "
            f"code for doing the M-step for the initialization parameters."
        )

    if example_end_times is None:
        example_end_times = np.array([-1, T])

    if not example_end_times_are_proper(example_end_times, len(continuous_states)):
        raise ValueError(
            f"Event end times do not have the proper format. Consult the `examples` module "
            f"and try again.  `example_end_times` MUST begin with -1 and end with T, the length "
            f"of the grand time series."
        )

    if system_covariates is None:
        # TODO: Check that D_s=0 as well; if not there is an inconsistency in the implied desire of the caller.
        system_covariates = np.zeros((T, 0))

    if DIMS.L == 1:
        # Automatically turn off M-step for system level parameters if there is only one system state.
        M_step_toggles.STP = M_Step_Toggle_Value.OFF

    # TODO:  I need to have a way to do a DUMB (default/non-data-informed) init for both VEZ and VES summaries
    # so that we can get ELBO baselines BEFORE the smart-initialization.... Maybe make VEZ, VES uniform? And
    # use the data-free inits for everything else?

    if verbose:
        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            example_end_times,
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
            example_end_times,
            system_covariates,
            use_continuous_states,
        )

        if verbose:
            print(f"\nVES step's log normalizer: {VES_summary.log_normalizer:.02f}")
            if true_system_regimes is not None:
                most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1)
                pct_correct_system = compute_regime_labeling_accuracy(most_likely_system_regimes, true_system_regimes)
                print(f"Percent correct classifications for system segmentations {pct_correct_system:.02f}")

        VEZ_summaries = run_VEZ_step_JAX(
            all_params.CSP,
            all_params.ETP,
            all_params.IP,
            continuous_states,
            VES_summary.expected_regimes,
            model,
            example_end_times,
        )

        if verbose:
            print(
                f"\nVEZ step's log normalizer by entities for continuous state emissions when we use VES inits for q(S): {VEZ_summaries.log_normalizers}"
            )
            if true_entity_regimes is not None:
                pct_corrects_entities = np.empty(DIMS.J)
                for j in range(DIMS.J):
                    most_likely_system_regimes = np.argmax(VEZ_summaries.expected_regimes[:, j, :], axis=1)
                    pct_corrects_entities[j] = compute_regime_labeling_accuracy(
                        most_likely_system_regimes, true_entity_regimes[:, j]
                    )
                print(f"Percent correct classifications for entity-level segmentations {pct_corrects_entities}")

            elbo_decomposed = compute_elbo_decomposed(
                all_params,
                VES_summary,
                VEZ_summaries,
                system_transition_prior,
                continuous_states,
                model,
                example_end_times,
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
            example_end_times,
            use_continuous_states,
            verbose,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            example_end_times,
            system_covariates,
        )
        if verbose:
            print(
                f"After ETP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        ###
        # M-step (STP)
        ###

        # Note the VES step has already taken care of the `use_continuous_states` mask.
        all_params = run_M_step_for_STP(
            all_params,
            M_step_toggles.STP,
            VES_summary,
            system_transition_prior,
            i,
            num_M_step_iters,
            model,
            example_end_times,
            system_covariates,
            continuous_states,
            verbose,
        )

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            example_end_times,
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
        example_end_times,
        use_continuous_states,
        )

        elbo_decomposed = compute_elbo_decomposed(
        all_params,
        VES_summary,
        VEZ_summaries,
        system_transition_prior,
        continuous_states,
        model,
        example_end_times,
        system_covariates,
        )
        if verbose:
            print(
            f"After CSP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

        ###
        # M-step (IP)
        ###

        IP_new = run_M_step_for_IP(
            all_params.IP,
            M_step_toggles.IP,
            VES_summary,
            VEZ_summaries,
            continuous_states,
            example_end_times,
        )
        all_params = AllParameters_JAX(all_params.STP, all_params.ETP, all_params.CSP, all_params.EP, IP_new)
        # TODO: Make all `run_M_step[...]` have consistent call signatures.
        # I think the current more compact one is better; otherwise we have to construct
        # a full all parameters instance (with lots of extraneous info) when all we want to do is an operation
        # on the initial params.

        elbo_decomposed = compute_elbo_decomposed(
            all_params,
            VES_summary,
            VEZ_summaries,
            system_transition_prior,
            continuous_states,
            model,
            example_end_times,
            system_covariates,
        )
        if verbose:
            print(
                f"After IP-M step on iteration {i+1}, we have Elbo: {elbo_decomposed.elbo:.02f}. Energy: {elbo_decomposed.energy:.02f}. Entropy: { elbo_decomposed.entropy:.02f}. "
            )

    return VES_summary, VEZ_summaries, all_params, elbo_decomposed
