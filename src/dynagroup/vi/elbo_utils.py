import functools
import warnings
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Union

import numpy as np
import jax.numpy as jnp
import jax_dataclasses as jdc

from dynagroup.model import Model
from dynagroup.params import (
    AllParameters_JAX,
    SystemTransitionParameters_JAX,
    EntityTransitionParameters_MetaSwitch_JAX,
    ContinuousStateParameters_JAX,
    InitializationParameters_JAX,
)
from dynagroup.vi.prior import SystemTransitionPrior_JAX

from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray5D,
    NumpyArray1D,
    NumpyArray2D,
)
from dynagroup.hmm_posterior import (
    HMM_Posterior_Summaries_JAX,
    HMM_Posterior_Summary_JAX,
)
from dynagroup.examples import (
    eligible_transitions_to_next,
    get_initialization_times,
    get_non_initialization_times,
)
from dynagroup.sticky import evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant
from dynagroup.util import normalize_log_potentials_by_axis_JAX

def calc_elbo(
    params: AllParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    STP_prior: Optional[SystemTransitionPrior_JAX],
    model: Model,
    data_TJD: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_TJ: Optional[JaxNumpyArray2D] = None,
    return_dict: bool = False,
    system_covariates: Optional[JaxNumpyArray2D]=None,
) -> float:
    elbo_init_dict = calc_energy__init_sys_init_entity_init_data_lik(
        params.IP, VES_summary, VEZ_summaries,
        model, data_TJD, example_end_times, mask_TJ, True)
    Elogp_s1toT = calc_energy__system_trans(
        params.STP, VES_summary,
        model, data_TJD, example_end_times, system_covariates)
    Elogp_s0toT = Elogp_s1toT + elbo_init_dict['Elogp_s0']
    assert Elogp_s0toT < 1e-7 # expected logpmf of discrete should be negative

    Elogp_z1toT = calc_energy__entity_trans(
        params.ETP, VES_summary, VEZ_summaries,
        model, data_TJD, example_end_times, mask_TJ)
    Elogp_z0toT = Elogp_z1toT + elbo_init_dict['Elogp_z0']
    assert Elogp_z0toT < 1e-7 # expected logpmf of discrete should be negative

    Elogp_x1toT = calc_energy__data_likelihood(
        params.CSP, VEZ_summaries,
        model, data_TJD, example_end_times, mask_TJ)
    Elogp_x0toT = Elogp_x1toT + elbo_init_dict['Elogp_x0']

    if STP_prior is not None:
        logpdf_prior_STP = evaluate_log_probability_density_of_sticky_transition_matrix_up_to_constant(
            normalize_log_potentials_by_axis_JAX(params.STP.Pi, axis=1),
            STP_prior.alpha,
            STP_prior.kappa,
        )
    else:
        logpdf_prior_STP = 0.0

    energy = Elogp_s0toT + Elogp_z0toT + Elogp_x0toT
    entropy_qs = jnp.sum(VES_summary.entropy) # ensures cast to jax
    entropy_qz = jnp.sum(VEZ_summaries.entropies)
    elbo = energy + entropy_qz + entropy_qs + logpdf_prior_STP
    if return_dict:
        return {
            'elbo':elbo,
            'energy':energy, 'entropy':entropy_qs + entropy_qz,
            'logpdf_prior_STP':logpdf_prior_STP,
            'entropy_qs': entropy_qs, 'entropy_qz':entropy_qz,
            'Elogp_s0toT':Elogp_s0toT, 'Elogp_z0toT':Elogp_z0toT,
            'Elogp_x0toT':Elogp_x0toT}
    return elbo

def calc_energy__init_sys_init_entity_init_data_lik(
    IP: InitializationParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    data_TJD: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_TJ: Optional[JaxNumpyArray2D] = None,
    return_dict: bool = False,
) -> float:
    init_times_V = get_initialization_times(example_end_times)
    V = len(init_times_V)
    J, K = np.shape(IP.pi_entities)

    Es_VL = VES_summary.expected_regimes[init_times_V]
    Ns_L = jnp.sum(Es_VL, axis=0)  # shape (L,)
    logprob_s0_1L = jnp.log(IP.pi_system)[None,:]
    Elogp_s0 = jnp.sum(Ns_L * logprob_s0_1L)

    Ez_TJK = VEZ_summaries.expected_regimes    
    Elogp_z0 = 0.0
    for j in range(J):
        Ezj_VK = Ez_TJK[init_times_V, j]
        Nj_K = jnp.sum(Ezj_VK, axis=0)  # shape (K,)
        Elogp_z0 += jnp.sum(Nj_K * jnp.log(IP.pi_entities[j]))

    Elogp_x0 = 0.0
    for t_init in init_times_V:
        logpdf_x_t_JK = model.compute_log_initial_continuous_state_emissions_JAX(
            IP, data_TJD[t_init]
         )
        Ez_JK = Ez_TJK[t_init]
        Elogp_x0 += jnp.sum(Ez_JK * logpdf_x_t_JK)

    if return_dict:
        return {'Elogp_s0':Elogp_s0, 'Elogp_z0':Elogp_z0, 'Elogp_x0':Elogp_x0}
    return Elogp_s0 + Elogp_z0 + Elogp_x0    


def calc_energy__system_trans(
    STP: SystemTransitionParameters_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    model: Model,
    data_TJD: Optional[JaxNumpyArray3D],
    example_end_times: NumpyArray1D,
    system_covariates: Optional[JaxNumpyArray2D]=None,
) -> float:
    """ Compute E[ log p(s_0:T | x, theta)]
    """
    # s_ULL has shape (T-1,L,L)
    # s_ULL[t,l,l'] := q(s_{t+1}=l', s_t=l)
    s_ULL = VES_summary.expected_joints
    T_minus_1 = np.shape(s_ULL)[0]

    # `log_transition_matrices` has shape (T-1,L,L)
    log_trans_prob_ULL = model.compute_log_system_transition_probability_matrices_JAX(
        STP,
        T_minus_1,
        system_covariates=system_covariates,
        x_prevs=data_TJD[:-1],
        system_recurrence_transformation=model.transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX,
    )
    # Create binary mask of which tsteps are eligible
    elig_bmask_U = eligible_transitions_to_next(example_end_times)
    s_MLL = s_ULL[elig_bmask_U]
    log_trans_MLL = log_trans_prob_ULL[elig_bmask_U]
    return jnp.sum(s_MLL * log_trans_MLL)


def calc_energy__entity_trans(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    data_TJD: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_TJ: Optional[JaxNumpyArray2D] = None,
) -> float:
    """ Compute sum_j E[ log p(z_0:T | s_0:T, x, theta)]
    Arguaments:
        continuous_states: has shape (T, J, D)
    """
    T, J, D = np.shape(data_TJD)
    U = T - 1
    # Compute E[ s_t+1=l, z_t = k, z_t+1 = k']
    Eszz_UJLKK = calc_prob_of_regime_triplets_at_adjacent_times_JAX(
        VES_summary, VEZ_summaries)
    # `log_transition_matrices` has shape (T-1,J,L,K,K)
    log_trans_UJLKK = model.compute_log_entity_transition_probability_matrices_JAX(
        ETP,
        data_TJD[:-1],
        model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    )
    # Mask out sequence boundaries
    elig_bmask_U = eligible_transitions_to_next(example_end_times)
    elig_bmask_U1111 = elig_bmask_U[:, None, None, None, None]
    # Also ignore parts of any entity sequence with no data
    mask_UJ111 = mask_TJ[1:, :, None, None, None]
    mask_UJ111 = jnp.logical_and(elig_bmask_U1111, mask_UJ111)
    return jnp.sum(Eszz_UJLKK * (log_trans_UJLKK * mask_UJ111))
    


def calc_energy__data_likelihood(
    CSP: ContinuousStateParameters_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    model: Model,
    data_TJD: JaxNumpyArray3D,
    example_end_times: NumpyArray1D,
    mask_TJ: Optional[JaxNumpyArray2D] = None,
) -> float:
    """ Compute sum_j E[ log p(x^j_0:T | all z, all s, \theta)]

    Arguments:
        use_continuous_states: Defaults to None (which means all states are used) because
            when we compute the ELBO, we always assume a full dataset.  This is simply because
            I haven't had the time yet to dig into ssm.messages to handle partial observations
            when doing forward backward.
    """
    T, J, D = np.shape(data_TJD)
    
    non_init_times_V = get_non_initialization_times(example_end_times)
    non_init_times_lower_index_V = non_init_times_V - 1
    V = len(non_init_times_V)

    # Compute probability of each non-init time t assigned to each state k
    Ez_VJK = VEZ_summaries.expected_regimes[non_init_times_V]

    # Compute likelihood of each data obs under each state k
    # We compute the AR likelihood of t=1 given t=0, t=2 given t=1, etc.
    # Thus, indices need to be shifted down by one
    logpdf_UJK = model.compute_log_continuous_state_emissions_after_initial_timestep_JAX(
        CSP, data_TJD)
    logpdf_VJK = logpdf_UJK[non_init_times_lower_index_V]

    # Ignore parts of any entity sequence with no data
    mask_VJ1 = mask_TJ[non_init_times_V, :, None]
    return jnp.sum(Ez_VJK * (mask_VJ1 * logpdf_VJK))


#######
## Utils
#######


def calc_prob_of_regime_triplets_at_adjacent_times_JAX(
    VES_summary: HMM_Posterior_Summary_JAX,
    VEZ_summaries: HMM_Posterior_Summaries_JAX,
    assert_valid_output: bool = True, # turn off to go fast
) -> JaxNumpyArray5D:
    """
    Returns:
        q_UJLKK : np.array of shape (T-1,J,L,K,K).  
            entry (t,j,l,k,k') := q( z^j_t = k, z^j_{t+1} = k', s^j_{t+1} = l)
            Probability of j-th entity transition, when time goes from t to t+1,
            transitioning from its regime k to k' under system state l

            Should define a valid distribution over all l, k, k' values.
    """
    sys_r_U1L11 = VES_summary.expected_regimes[1:, None, :, None, None]
    ent_s_UJ1KK = VEZ_summaries.expected_joints[:, :, None, :, :]
    q_UJLKK = sys_r_U1L11 * ent_s_UJ1KK
    if assert_valid_output:
        assert np.allclose(jnp.sum(q_UJLKK,axis=(2,3,4)), 1.)
    return q_UJLKK
