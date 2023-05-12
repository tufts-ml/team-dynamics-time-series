from dataclasses import dataclass
from typing import List, Optional, Union

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from ssm.messages import hmm_expected_states

from dynagroup.types import (
    JaxNumpyArray1D,
    JaxNumpyArray2D,
    JaxNumpyArray3D,
    JaxNumpyArray4D,
    NumpyArray1D,
    NumpyArray2D,
    NumpyArray3D,
    NumpyArray4D,
)
from dynagroup.util import normalize_potentials_by_axis, soften_tpm


###
# Structs
###


@dataclass
class HMM_Posterior_Summary_NUMPY:
    """
    Gives posterior summary for a "HMM", by which we mean a model
    where some discrete latent chain, x_{1:T}, generates some
    emissions chain, y_{1:T}, in a way such that
        p(y_t | past) :=  p(y_t | x_{1:t}, y_{1:t-1}) = p(y_t|x_t)

    The posterior summary tells us what we need to know about
        p(x_{1:T} | y_{1:T})

    Attributes:
        expected_regimes: np.array with shape (T,K)
            Gives E[x_t | y_{1:T}]
        expected_joints: np.array with shape (T-1, K, K)
            Gives E[x_{t+1}, x_t | y_{1:T}]; that is, the (t,k,k')-th element gives
            the probability distribution over all pairwise options
            (x_{t+1}=k', x_{t}=k | y_{1:T}) for t=1,...t-1
        log_normalizer : float
            The log probability density over the emissions chain (y_{1:T}),
            which can be obtained by marginalziing the last filtered joint p(x_T, y_{1:T})
            over the probabilities for each final latent variable x_T.
        entropy: optional float

    Notation:
        T: number of timesteps
        K: number of regimes
    """

    expected_regimes: NumpyArray2D
    expected_joints: NumpyArray3D
    log_normalizer: float
    entropy: Optional[float] = None


@jdc.pytree_dataclass
class HMM_Posterior_Summary_JAX:
    """
    Gives posterior summary for a "HMM", by which we mean a model
    where some discrete latent chain, x_{1:T}, generates some
    emissions chain, y_{1:T}, in a way such that
        p(y_t | past) :=  p(y_t | x_{1:t}, y_{1:t-1}) = p(y_t|x_t)

    The posterior summary tells us what we need to know about
        p(x_{1:T} | y_{1:T})

    Attributes:
        expected_regimes: np.array with shape (T,K)
            Gives E[x_t | y_{1:T}]
        expected_joints: np.array with shape (T-1, K, K)
            Gives E[x_{t+1}, x_t | y_{1:T}]; that is, the (t,k,k')-th element gives
            the probability distribution over all pairwise options
            (x_{t+1}=k', x_{t}=k | y_{1:T}) for t=1,...t-1
        log_normalizer : float
            The log probability density over the emissions chain (y_{1:T}),
            which can be obtained by marginalziing the last filtered joint p(x_T, y_{1:T})
            over the probabilities for each final latent variable x_T.
        entropy: optional float

    Notation:
        T: number of timesteps
        K: number of regimes
    """

    expected_regimes: JaxNumpyArray2D
    expected_joints: JaxNumpyArray3D
    log_normalizer: float
    entropy: Optional[float] = None


HMM_Posterior_Summary = Union[HMM_Posterior_Summary_JAX, HMM_Posterior_Summary_NUMPY]


@dataclass
class HMM_Posterior_Summaries_NUMPY:
    """
    WARNING:
        This class is, as of now, not currently instantiated when running CAVI with numpy.
        That uses a List[HMM_Posterior_Summary].  This class is for post-inference conversions
        from HMM_Posterior_Summaries_JAX, so that we can do things like write diagnostic code
        with for loops instead of the horrible vectorized code.

    Overview:
        Gives posterior summaries for J "HMM"s, by which we mean a model
        where some discrete latent chain, x_{1:T}^(j), generates some
        emissions chain, y_{1:T}^(j), in a way such that
            p(y_t^j | past^j) :=  p(y_t | x_{1:t}^j, y_{1:t-1}^j) = p(y_t^j|x_t^j)

        The posterior summary tells us what we need to know about
            p(x_{1:T}^j | y_{1:T}^j)
        for each j=1,..,J

    Attributes:
        expected_regimes: np.array with shape (T,J,K)
            Gives E[x_t^j | y_{1:T}^j]
        expected_joints: np.array with shape (T-1, J, K, K)
            Gives E[x_{t+1}^j, x_t^j | y_{1:T}^j]; that is, the (t,j,k,k')-th element gives
            the probability distribution over all pairwise options
            (x_{t+1}^j=k', x_{t}^j=k | y_{1:T}^j).
        log_normalizers : np.array with shape (J,)
            The log probability density over the emissions chain (y_{1:T}^j),
            which can be obtained by marginalziing the last filtered joint p(x_T^j, y_{1:T}^j)
            over the probabilities for each final latent variable x_T^j.
        entropies: optional np.array with shape(J,)

    Notation:
        T: number of timesteps
        K: number of regimes
        J: number of entities
    """

    expected_regimes: NumpyArray3D
    expected_joints: NumpyArray4D
    log_normalizers: NumpyArray1D
    entropies: Optional[NumpyArray1D] = None


@jdc.pytree_dataclass
class HMM_Posterior_Summaries_JAX:
    """
    Gives posterior summaries for J "HMM"s, by which we mean a model
    where some discrete latent chain, x_{1:T}^(j), generates some
    emissions chain, y_{1:T}^(j), in a way such that
        p(y_t^j | past^j) :=  p(y_t | x_{1:t}^j, y_{1:t-1}^j) = p(y_t^j|x_t^j)

    The posterior summary tells us what we need to know about
        p(x_{1:T}^j | y_{1:T}^j)
    for each j=1,..,J

    Attributes:
        expected_regimes: np.array with shape (T,J,K)
            Gives E[x_t^j | y_{1:T}^j]
        expected_joints: np.array with shape (T-1, J, K, K)
            Gives E[x_{t+1}^j, x_t^j | y_{1:T}^j]; that is, the (t,j,k,k')-th element gives
            the probability distribution over all pairwise options
            (x_{t+1}^j=k', x_{t}^j=k | y_{1:T}^j).
        log_normalizers : np.array with shape (J,)
            The log probability density over the emissions chain (y_{1:T}^j),
            which can be obtained by marginalziing the last filtered joint p(x_T^j, y_{1:T}^j)
            over the probabilities for each final latent variable x_T^j.
        entropies: optional np.array with shape(J,)

    Notation:
        T: number of timesteps
        K: number of regimes
        J: number of entities
    """

    expected_regimes: JaxNumpyArray3D
    expected_joints: JaxNumpyArray4D
    log_normalizers: JaxNumpyArray1D
    entropies: Optional[JaxNumpyArray1D] = None


###
# Compute entropy
###


def compute_entropy_of_HMM_posterior(
    log_transitions: JaxNumpyArray3D,
    log_emissions: JaxNumpyArray2D,
    log_init: JaxNumpyArray1D,
    posterior_summary_without_entropy: HMM_Posterior_Summary_JAX,
):
    """
    Compute the entropy of an HMM posterior.

    The HMM is a joint probability distribution over (x,y) := (x_{1:T}, y_{1:T}),
    where each x_t \in {1,...,K} are the discrete latent states and y_t are the observations

    The entropy of the posterior is given by
    H(p(x|y)) = - \sum_{k=1}^K p(x_1=k |y) log p(x_1=k)
                - \sum_{t=2}^T \sum_{k=1}^K \sum_{k'=1}^K p(x_t=k', x_{t-1}=k | y) log p(x_t=k'| x_{t-1}=k)
                - \sum_{t=1}^T \sum_{k=1}^K p(x_t | y) log p(y_t | x_t=k)
                - log p(y)

    Arguments:
        log_init: An array of shape (K,)
            whose k-th entry gives the MODEL's log init state probability
            p(x_1 =k)
        log_transitions: An array of shape (T-1,K,K),
            whose (t,k,k')-th entry gives the MODEL's log probability
            of p(x_t = k' | x_{t-1}=k)
        log_emissions : An array of shape (T,K),
            whose (t,k)-th entry gives the MODEL's log emissions density
            of p(y_t | x_t=k)
        posterior_summary:  Has attributes
            - log_normalizer : float
            - expected_regimes: array of shape (T,K)
            - expected_joints: array of shape (T-1,K,K)
    """
    PS = posterior_summary_without_entropy
    entropy = 0.0
    entropy += PS.log_normalizer  # float
    entropy += -jnp.sum(PS.expected_regimes * log_emissions)
    entropy += -jnp.sum(PS.expected_joints * log_transitions)
    entropy += -jnp.sum(PS.expected_regimes[0] * log_init)
    return entropy


###
# Compute posterior summary/summaries
###


def compute_hmm_posterior_summary_NUMPY(
    log_transitions: JaxNumpyArray3D,
    log_emissions: JaxNumpyArray2D,
    init_dist_over_regimes: JaxNumpyArray1D,
) -> HMM_Posterior_Summary_NUMPY:
    transitions = np.exp(log_transitions)
    expected_regimes, expected_joints, log_normalizer = hmm_expected_states(
        init_dist_over_regimes,
        transitions,
        log_emissions,
    )

    hmm_posterior_summary_without_entropy = HMM_Posterior_Summary_NUMPY(
        expected_regimes,
        expected_joints,
        log_normalizer,
        entropy=None,
    )

    log_init = np.log(init_dist_over_regimes)
    entropy = compute_entropy_of_HMM_posterior(
        log_transitions,
        log_emissions,
        log_init,
        hmm_posterior_summary_without_entropy,
    )
    return HMM_Posterior_Summary_NUMPY(expected_regimes, expected_joints, log_normalizer, entropy)


def compute_hmm_posterior_summary_JAX(
    log_transitions: JaxNumpyArray3D,
    log_emissions: JaxNumpyArray2D,
    init_dist_over_regimes: JaxNumpyArray1D,
) -> HMM_Posterior_Summary_JAX:
    """
    Arguments:
        log_transitions: An array of shape (T-1,K,K),
            whose (t,k,k')-th entry gives the MODEL's log probability
            of p(x_t = k' | x_{t-1}=k)
        log_emissions : An array of shape (T,K),
            whose (t,k)-th entry gives the MODEL's log emissions density
            of p(y_t | x_t=k)
        init_dist_over_regimes: An array of shape (K,)
            whose k-th entry gives the MODEL's init state probability
            p(x_1 =k)
    """
    transitions = jnp.exp(log_transitions)

    # TODO: Here is where we lose jax-ness...Questions
    # 1) Does the conversion to numpy slow things down?
    # 2) Should we convert the return values to jax arrays?
    # 3) Is there a function in dynamax which does hmm_expected_states but with jax i/o?
    expected_regimes, expected_joints, log_normalizer = hmm_expected_states(
        np.asarray(init_dist_over_regimes, dtype=np.float64),
        np.asarray(transitions, dtype=np.float64),
        np.asarray(log_emissions, dtype=np.float64),
    )

    # ### RK: I tried running the corrresponding dynamax function,  so we don't have to convert to jax and back,
    # ### but their dynamax funtion seems to be dropping a time-step for expected_joints in the setting where
    # ### there are time-dependent parameters.
    # ### See https://github.com/probml/dynamax/issues/310.
    # ### TODO: The problem is that trans_probs is one timestep too short! check how dynamax does this!
    # from dynamax.hidden_markov_model import hmm_smoother
    # result=hmm_smoother(init_dist_over_system_regimes, transitions, log_emissions)
    # expected_regimes, expected_joints, log_normalizer  =  result.smoothed_probs, result.trans_probs, float(result.marginal_loglik)

    hmm_posterior_summary_without_entropy = HMM_Posterior_Summary_JAX(
        jnp.asarray(expected_regimes),
        jnp.asarray(expected_joints),
        jnp.asarray(log_normalizer),
        entropy=None,
    )

    log_init = jnp.log(init_dist_over_regimes)
    entropy = compute_entropy_of_HMM_posterior(
        log_transitions,
        log_emissions,
        log_init,
        hmm_posterior_summary_without_entropy,
    )
    return HMM_Posterior_Summary_JAX(
        jnp.asarray(expected_regimes),
        jnp.asarray(expected_joints),
        jnp.asarray(log_normalizer),
        entropy,
    )


def compute_hmm_posterior_summaries_JAX(
    log_transitions: JaxNumpyArray4D,
    log_emissions: JaxNumpyArray3D,
    init_dists_over_regimes: JaxNumpyArray2D,
) -> HMM_Posterior_Summaries_JAX:
    """
    Arguments:
        log_transitions: An array of shape (T-1,J,K,K),
            whose (t,j,k,k')-th entry gives the MODEL's log probability
            of p(x_t^j = k' | x_{t-1}^j=k) for t=2,...,T
        log_emissions : An array of shape (T,J,K),
            whose (t,j,k)-th entry gives the MODEL's log emissions density
            of p(y_t^j | x_t^j=k)
        init_dists_over_regimes: np.array of size (J,K)
            The j-th row must live on the simplex for all j=1,...,J.
    """
    # first create LISTS, where the j-th element of each list
    # gives an attribute from the hmm posterior summaries (including entropy)
    J = jnp.shape(log_transitions)[1]
    (
        expected_regimes_list,
        expected_joints_list,
        log_normalizers_list,
        entropies_list,
    ) = ([None] * J, [None] * J, [None] * J, [None] * J)
    for j in range(J):
        hmm_summary_for_entity = compute_hmm_posterior_summary_JAX(
            log_transitions[:, j], log_emissions[:, j], init_dists_over_regimes[j]
        )
        expected_regimes_list[j] = hmm_summary_for_entity.expected_regimes
        expected_joints_list[j] = hmm_summary_for_entity.expected_joints
        log_normalizers_list[j] = hmm_summary_for_entity.log_normalizer
        entropies_list[j] = hmm_summary_for_entity.entropy

    # then convert these to the right shape for the HMM_Posterior_Summaries_JAX class
    (
        expected_regimes_entity_first,
        expected_joints_entity_first,
        log_normalizers,
        entropies,
    ) = (
        jnp.asarray(expected_regimes_list),
        jnp.asarray(expected_joints_list),
        jnp.asarray(log_normalizers_list),
        jnp.asarray(entropies_list),
    )
    expected_regimes = jnp.swapaxes(expected_regimes_entity_first, 0, 1)  # (T,J,K)
    expected_joints = jnp.swapaxes(expected_joints_entity_first, 0, 1)  # (T-1,J,K,K)

    return HMM_Posterior_Summaries_JAX(
        expected_regimes, expected_joints, log_normalizers, entropies
    )


def compute_hmm_posterior_summaries_NUMPY(
    log_transitions: NumpyArray4D,
    log_emissions: NumpyArray3D,
    init_dists_over_regimes: NumpyArray2D,
) -> List[HMM_Posterior_Summary_NUMPY]:
    """
    Arguments:
        log_transitions: An array of shape (T-1,J,K,K),
            whose (t,j,k,k')-th entry gives the MODEL's log probability
            of p(x_t^j = k' | x_{t-1}^j=k)
        log_emissions : An array of shape (T,J,K),
            whose (t,j,k)-th entry gives the MODEL's log emissions density
            of p(y_t^j | x_t^j=k)
        init_dists_over_regimes: np.array of size (J,K)
            The j-th row must live on the simplex for all j=1,...,J.
    """
    J = np.shape(log_transitions)[1]
    hmm_posterior_summaries = [None] * J
    for j in range(J):
        hmm_posterior_summaries[j] = compute_hmm_posterior_summary_NUMPY(
            log_transitions[:, j], log_emissions[:, j], init_dists_over_regimes[j]
        )
    return hmm_posterior_summaries


###
# Convert jax to numpy and back
###


def convert_hmm_posterior_summaries_from_jax_to_numpy(
    hmm_posterior_summaries: HMM_Posterior_Summaries_JAX,
) -> HMM_Posterior_Summaries_NUMPY:
    entropies = (
        None
        if hmm_posterior_summaries.entropies is None
        else np.asarray(hmm_posterior_summaries.entropies)
    )
    return HMM_Posterior_Summaries_NUMPY(
        np.asarray(hmm_posterior_summaries.expected_regimes),
        np.asarray(hmm_posterior_summaries.expected_joints),
        np.asarray(hmm_posterior_summaries.log_normalizers),
        entropies,
    )


###
# Produce closed-form M-step
###


# TODO: Is there some way to combine `compute_closed_form_M_step`
# with `compute_closed_form_M_step_on_posterior_summaries` by just vectorizing across
# any leading dimensions when they exist?
def compute_closed_form_M_step(posterior_summary: HMM_Posterior_Summary_NUMPY) -> NumpyArray2D:
    """
    Returns:
        Array of shape (K,K) which is a tpm.
    """
    K = np.shape(posterior_summary.expected_regimes)[1]

    # Compute tpm
    tpm_empirical = np.zeros((K, K))
    for k in range(K):
        for k_prime in range(K):
            tpm_empirical[k, k_prime] = np.sum(
                posterior_summary.expected_joints[:, k, k_prime], axis=0
            ) / np.sum(posterior_summary.expected_regimes[:-1, k], axis=0)

    # Add in a small bit of a uniform distribution to bound away from exact ones and zeros.
    # A better approach is to use a Dirichlet prior and take the posterior.
    return soften_tpm(tpm_empirical)


# TODO: Is there some way to combine `compute_closed_form_M_step`
# with `compute_closed_form_M_step_on_posterior_summaries` by just vectorizing across
# any leading dimensions when they exist?


def compute_closed_form_M_step_on_posterior_summaries(
    posterior_summaries: HMM_Posterior_Summaries_NUMPY,
    observation_weights: Optional[NumpyArray2D] = None,
) -> NumpyArray3D:
    """
    Arguments:
        observation_weights: If None, we assume all states were observed.
            Otherwise, this is a (T,J) binary vector such that
            the (t,j)-th element  is 1 if continuous_states[t,j] was observed
            and 0 otherwise.  For any (t,j) that wasn't observed, we don't use
            that info to do the M-step.

    Returns:
        Array of shape (J,K,K), whose j-th entry is a tpm
    """

    T, J, K = np.shape(posterior_summaries.expected_regimes)

    if observation_weights is None:
        observation_weights = np.ones((T, J))

    # Compute tpm
    empirical_tpms_softened = np.zeros((J, K, K))
    for j in range(J):
        expected_joints_for_observed_data = (
            posterior_summaries.expected_joints[:, j] * observation_weights[:, j][:-1, None, None]
        )
        expected_regimes_for_observed_data = np.sum(expected_joints_for_observed_data, axis=-2)
        # alternatively, we could do posterior_summaries.expected_regimes[1:,j] =  expected_regimes_for_observed_data ,
        # but this way we only need to take care of the observation_weights once.

        from_to_sums = np.sum(expected_joints_for_observed_data, axis=0)
        from_sums = np.sum(expected_regimes_for_observed_data, axis=0)

        empirical_tpm = from_to_sums / from_sums[:, np.newaxis]

        # TODO: Check this and confirm that it's doing the M-step correctly.  I'm not getting exact sums to 1
        # (instead it's like .996), but this could be because of float32

        # Add in a small bit of a uniform distribution to bound away from exact ones and zeros.
        # A better approach is to use a Dirichlet prior and take the posterior.
        empirical_tpms_softened[j] = normalize_potentials_by_axis(soften_tpm(empirical_tpm), axis=1)
    return empirical_tpms_softened
