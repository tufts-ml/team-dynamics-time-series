import numpy as np
import numpy.random as npr
from scipy.special import logsumexp

from dynagroup.params import (
    AllParameters,
    ContinuousStateParameters_Gaussian,
    EmissionsParameters,
    EntityTransitionParameters_MetaSwitch,
    InitializationParameters,
    SystemTransitionParameters,
)
from dynagroup.sticky import sample_sticky_transition_matrix
from dynagroup.types import NumpyArray1D, NumpyArray2D
from dynagroup.util import generate_random_covariance_matrix, random_rotation


np.set_printoptions(suppress=True, precision=3)


"""
Model 1 is the model with "top-level" recurrence from entity regimes to system regimes.
Model 2 is the top-down meta-switching model.
Model 2a refers to the fact that here we'll take the x's to be observed.
"""


###
# One step ahead transitions
###

# TODO: Need to rewrite the generic Model 2a functions so that they come from the
# Model Factors; this is the new way of using sample_team_dynamics.  For an example,
# see the Figure8 code.


def log_probs_for_one_step_ahead_system_transitions(
    STP: SystemTransitionParameters,
    prev_entity_regimes: NumpyArray1D,
    prev_system_regime: int,
) -> NumpyArray1D:
    """
    Attributes:
        prev_entity_regimes :  has shape (J,)
            Each entry is in {1,...,K}
            Gives values of zs (entity regimes) for each entity j at previous timestep
        prev_system_regime  : int
            In {1,...,L}.
            Gives value of s (system regime) at previous timestep
    Returns:
        Probability vector over {1,...,L}
    """

    # TODO : Allow covariates

    J, L, _ = np.shape(STP.Gammas)

    # contrib from prev entity regimes
    log_probs = np.zeros(L)
    for j in range(J):
        log_probs += STP.Gammas[j, :, prev_entity_regimes[j]]
    log_probs /= J

    # contrib from prev system regime
    log_probs += STP.Pi[prev_system_regime, :]

    # normalize
    log_probs -= logsumexp(log_probs)
    return log_probs


def log_probs_for_one_step_ahead_entity_transitions(
    ETP: EntityTransitionParameters_MetaSwitch,
    prev_entity_regimes: NumpyArray1D,
    prev_states: NumpyArray2D,
    curr_system_regime: int,
) -> NumpyArray2D:
    """
    Attributes:
        prev_entity_regimes :  has shape (J,)
            Each entry is in {1,...,K}
            Gives values of zs (entity regimes) for each entity j at previous timestep
        prev_states :  has shape (J,D)
            Gives values of continuous latent state, x in R^D,
            for each entity j at previous timestep
        curr_system_regime  : int
            In {1,...,L}.
            Gives value of s (system regime) at previous timestep
    Returns:
        np.array of shape (J,K)
            The j-th row gives a probability vector over {1,...,K}
    """

    # TODO : Allow covariates

    J, L, K, _ = np.shape(ETP.Psis)

    log_probs_unnormalized = np.zeros((J, K))
    for j in range(J):
        log_probs_unnormalized[j] += ETP.Psis[j, curr_system_regime] @ prev_states[j]
        log_probs_unnormalized[j] += ETP.Ps[j, curr_system_regime, prev_entity_regimes[j], :]

    # normalize
    log_normalizers = logsumexp(log_probs_unnormalized, axis=1)[:, None]
    log_probs = log_probs_unnormalized - log_normalizers

    return log_probs


###
# Configs
###

SEED = 1
J = 3
K, L = 4, 3
D = 4
N = 10
T = 500
M_s, M_e = 0, 0  # number of covariates for system and entity

# stickiness parameters
alpha_system, kappa_system = 1.0, 50.0
alpha_entity, kappa_entity = 1.0, 50.0

# dynamics parameters
a_scalar = 0.99
q_var = 0.1

###
# Initialization
###

np.random.seed(SEED)

# System Transition Parameters
exp_Pi = sample_sticky_transition_matrix(L, alpha=alpha_system, kappa=kappa_system, seed=SEED)
Pi = np.log(exp_Pi)
Gammas = np.zeros((J, L, K))  # Gammas must be zero for no feedback.
Upsilon = np.zeros((L, M_s))
STP = SystemTransitionParameters(Gammas, Upsilon, Pi)

# Entity Transition Parameters
Ps = np.zeros((J, L, K, K))
for j in range(J):
    exp_Ps = [
        sample_sticky_transition_matrix(
            K, alpha=alpha_entity, kappa=kappa_entity, seed=SEED + j + 1
        )
        for ell in range(L)
    ]
    Ps[j] = np.array([np.log(exp_P) for exp_P in exp_Ps])
Psis = npr.randn(J, L, K, D)
Omegas = np.zeros((J, L, K, M_e))
ETP = EntityTransitionParameters_MetaSwitch(Psis, Omegas, Ps)

# Continuous State Parameters
# As = npr.randn(J, K, D, D)
# TODO: Need to figure out how to make As stable etc.
As = np.zeros((J, K, D, D))
for j in range(J):
    for k in range(K):
        # TODO: randomly draw a_scalar (perhaps <=1.0) for each j,k
        As[j, k] = a_scalar * random_rotation(D, theta=np.pi / 20)

# generate_random_covariance_matrix(dim, var=1.0)

# bs = npr.randn(J, K, D)
bs = np.zeros((J, K, D))
# Qs = ss.invwishart(df=D, scale=np.eye(D)).rvs((J, K))
Qs = np.zeros((J, K, D, D))
for j in range(J):
    for k in range(K):
        Qs[j, k] = generate_random_covariance_matrix(dim=D, var=q_var)

CSP = ContinuousStateParameters_Gaussian(As, bs, Qs)

# Emissions Parameters
Cs = npr.randn(J, N, D)
# ds = npr.randn(J, N)
ds = np.zeros((J, N))

# Rs = ss.invwishart(df=N, scale=np.eye(N)).rvs(J)
Rs = np.zeros((J, N, N))
for j in range(J):
    Rs[j] = generate_random_covariance_matrix(dim=N, var=1.0)

EP = EmissionsParameters(Cs, ds, Rs)

# Initialization Parameters
pi_system = np.ones(L) / L
pi_entities = np.ones((J, K)) / K
mu_0s = npr.randn(J, K, D)
Sigma_0s = np.broadcast_to(np.eye(D), (J, K, D, D))
IP = InitializationParameters(pi_system, pi_entities, mu_0s, Sigma_0s)

# All Parameters
ALL_PARAMS = AllParameters(STP, ETP, CSP, EP, IP)

# ###
# # Make sample
# ###

# from dynagroup.sampler import sample_team_dynamics
#
# sample = sample_team_dynamics(
#     ALL_PARAMS,
#     T,
#     log_probs_for_one_step_ahead_system_transitions,
#     log_probs_for_one_step_ahead_entity_transitions,
#     seed=SEED,
# )

# # check that sample is "interesting" (e.g. non constant)
# # if it's not, initialize parameters with "strength" -- N(0,alpha)
# # instead of N(0,1)

# ###
# # Plot sample
# ###

# if __name__ == "__main__":
#     from dynagroup.plotting.sampling import plot_sample_with_system_regimes

#     for j in range(J):
#         print(f"Now plotting results for entity {j}")
#         plot_sample_with_system_regimes(
#             sample.xs[:, j, :], sample.ys[:, j, :], sample.zs[:, j], sample.s
#         )
