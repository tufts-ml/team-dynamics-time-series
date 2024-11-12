import numpy as np
import numpy.random as npr

from dynagroup.params import (
    AllParameters,
    ContinuousStateParameters_Gaussian,
    EmissionsParameters,
    EntityTransitionParameters_MetaSwitch,
    InitializationParameters_Gaussian,
    SystemTransitionParameters,
)
from dynagroup.sampler import sample_team_dynamics
from dynagroup.sticky import sample_sticky_transition_matrix
from dynagroup.types import NumpyArray4D
from dynagroup.util import generate_random_covariance_matrix
from dynagroup.util import ( # it's not random in 2dim.
    random_rotation as make_rotation_matrix,
)

from setup_model import figure8_model_JAX

np.set_printoptions(suppress=True, precision=3)

###
# Parameter Construction
###

def make_Psis_for_figure_8_experiment(
    J: int,
    recurrence_weight_to_entity_regime_favored_by_system: float,
    recurrence_weight_to_entity_regime_anti_favored_by_system: float,
):
    """
    Psis are the recurrence matrices for the entity-level transitions.

    Example:

        If recurrence_weight_to_entity_regime_favored_by_system  = 0.1
        and recurrence_weight_to_entity_regime_anti_favored_by_system  = -0.1,
        then Psis[j,:,:] for each j should be a LxK matrix of the form

        array([[ 0.1, -0.1],
               [-0.1,  0.1]])

        i.e., the weights favor entity regimes k such that k=ell.

    Notation:
        J: number of entities
        L: number of system-level regimes
        K: number of entity-level regimes
        D: dimensionality of latent continuous state, x
        D_e: dimensionality of latent continuous state, x, after applying transformation f.
    """
    L, K, D_e = 2, 2, 1  # these are fixed.
    Psis = np.zeros((J, L, K, D_e))

    # The D dimension is not needed here; we populate it redundantly to preserve the data generation format
    # from the generic data generation.
    for j in range(J):
        for l in range(L):
            for k in range(K):
                if k == l:
                    Psis[j, l, k, :] = recurrence_weight_to_entity_regime_favored_by_system
                else:
                    Psis[j, l, k, :] = recurrence_weight_to_entity_regime_anti_favored_by_system
    return Psis


def make_Ps_for_figure_8_experiment(J: int, entity_level_self_transition_prob: float):
    """
    Ps are the endogenous transition biases for the entity-level transitions.
    """
    L, K = 2, 2  # these are fixed.
    Ps = np.zeros((J, L, K, K))
    for j in range(J):
        for l in range(L):
            p = entity_level_self_transition_prob
            exp_P = np.array([[p, 1 - p], [1 - p, p]])
            Ps[j, l] = np.log(exp_P)
    return Ps


def make_bs_for_figure_8_experiment(
    J: int,
    As: NumpyArray4D,
):
    """
    We want rotations around a fix point in x space.
    We can write this as A(x-b) + b.
    Then by algebra, we have
         A(x-b) + b = Ax + (I-A)b
    Which decomposes the expression into a state matrix and a bias term.
    """
    K, D = 2, 2  # these are fixed.
    bs = np.zeros((J, K, D))
    for j in range(J):
        for k in range(K):
            if k == 0:
                circle_center = np.array([0, 1])  # in first entity regime, x rotation is centered around point (0,1)
            else:
                circle_center = np.array([0, -1])  # in first entity regime, x rotation is centered around point (0,1)
            bs[j, k] = (np.eye(D) - As[j, k]) @ circle_center
    return bs


def make_initialization_parameters_for_figure_8_experiment(
    J: int,
):
    # for figure 8 we force the initial x to be at (1,1),
    # the initial entity regime is the first one.
    # the initial state regime is the first one.
    D, L, K = 2, 2, 2  # these are fixed.

    prob_favored_regime = 0.999
    prob_unfavored_regimes = 1.0 - prob_favored_regime

    # set pi_system that we very likely always start at the first regime.
    pi_system = np.array([prob_favored_regime] + [prob_unfavored_regimes / (L - 1)] * (L - 1))

    # set pi_entities so that we very likely always start at the first regime.
    pi_entities = np.zeros((J, K))
    for j in range(J):
        pi_entities[j] = np.array([prob_favored_regime] + [prob_unfavored_regimes / (K - 1)] * (K - 1))

    # set mu_0's so that initial x will be at (1,1) with very high probability
    mu_0s = np.zeros((J, K, D))
    for j in range(J):
        for k in range(K):
            mu_0s[j, k, :] = np.array([1, 1])

    # set Sigma_0's to have small variance
    EPSILON = 1e-4
    Sigma_0s = np.broadcast_to(np.eye(D) * EPSILON, (J, K, D, D))

    return InitializationParameters_Gaussian(pi_system, pi_entities, mu_0s, Sigma_0s)


###
# Configs
###

### Fixed Configs
D = 2  # D MUST be 2 for Figure 8 toy dataset
L = 2  # L currently fixed to 2 for Figure 8 toy dataset so system states map bijectively to entity states
K = 2  # K currently fixed to 2 for Figure 8 toy dataset so system states map bijectively to entity states

#### Free Configs
SEED = 10
J = 3
N = 5
T = 400
M_e = 0  # number of covariates for entity
D_s = 0  # dimensionality of contribution to system-level transitions from skip-level recurrence and system-level covariates

# system transition parameters (stickiness)
alpha_system, kappa_system = 1.0, 50.0  # only needed if fixed_system_regimes=None
fixed_system_regimes = np.array([0] * int(T / 4) + [1] * int(T / 4) + [0] * int(T / 4) + [1] * int(T / 4))
# TODO: derive `idxs_of_system_regime_transitions` and `idxs_of_system_regime_transitions`
# from `fixed_system_regimes` programtically
times_of_system_regime_changepoints = [int(x) for x in np.arange(1, 4) * T / 4]
system_regime_ids_before_changepoints = [0, 1, 0, 1]

# entity transition parameters
entity_level_self_transition_prob = 0.999

# dynamics parameters
a_scalar = 1.0
q_var = 0.0001
periods_for_entities = [5, 20, 40]  # [10, 20, 30]

# entity transition parameters (special for figure 8)
recurrence_weight_to_entity_regime_favored_by_system = 2
recurrence_weight_to_entity_regime_anti_favored_by_system = -2

###
# Initialization
###

np.random.seed(SEED)

# System Transition Parameters
exp_Pi = sample_sticky_transition_matrix(L, alpha=alpha_system, kappa=kappa_system, seed=SEED)
Pi = np.log(exp_Pi)
Gammas = np.zeros((J, L, K))  # Gammas must be zero for no feedback.
Upsilon = np.zeros((L, D_s))
STP = SystemTransitionParameters(Gammas, Upsilon, Pi)

# Entity Transition Parameters
Ps = make_Ps_for_figure_8_experiment(J, entity_level_self_transition_prob)
Psis = make_Psis_for_figure_8_experiment(
    J,
    recurrence_weight_to_entity_regime_favored_by_system,
    recurrence_weight_to_entity_regime_anti_favored_by_system,
)
Omegas = np.zeros((J, L, K, M_e))
ETP = EntityTransitionParameters_MetaSwitch(Psis, Omegas, Ps)

# Continuous State Parameters
# As = npr.randn(J, K, D, D)
# TODO: Need to figure out how to make As stable etc.
if D != 2:
    raise ValueError("D must equal 2.")
As = np.zeros((J, K, D, D))
for j in range(J):
    # TODO: randomly draw a_scalar (perhaps <=1.0) for each j,k
    As[j, 0] = a_scalar * make_rotation_matrix(D, theta=-2 * np.pi / periods_for_entities[j])
    As[j, 1] = a_scalar * make_rotation_matrix(D, theta=2 * np.pi / periods_for_entities[j])
# Now bs
bs = make_bs_for_figure_8_experiment(J, As)
# Qs = ss.invwishart(df=D, scale=np.eye(D)).rvs((J, K))
Qs = np.zeros((J, K, D, D))
for j in range(J):
    for k in range(K):
        Qs[j, k] = generate_random_covariance_matrix(dim=D, var=q_var)

CSP = ContinuousStateParameters_Gaussian(As, bs, Qs)

# Emissions Parameters
Cs = npr.randn(J, N, D)
ds = np.zeros((J, N))

# Rs = ss.invwishart(df=N, scale=np.eye(N)).rvs(J)
Rs = np.zeros((J, N, N))
for j in range(J):
    Rs[j] = generate_random_covariance_matrix(dim=N, var=1.0)

EP = EmissionsParameters(Cs, ds, Rs)

# Initialization Parameters
# for figure 8 we force the initial x to be at (1,1),
# the initial entity regime is the first one.
# the initial state regime is the first one.
IP = make_initialization_parameters_for_figure_8_experiment(J)

# Package up all parameters
TRUE_PARAMS = AllParameters(STP, ETP, CSP, EP, IP)

###
# Make sample that is fully observed for all entities
###

dataset = sample_team_dynamics(
    TRUE_PARAMS,
    T,
    model=figure8_model_JAX,
    seed=SEED,
    fixed_system_regimes=fixed_system_regimes,
)

def find_last_index_in_interval_where_array_value_is_close_to_desired_point(
    array: np.array,
    desired_point: np.array,
    starting_index: int,
    ending_index: int,
) -> float:
    closeness_threshold = 0.2

    for t in reversed(range(starting_index, ending_index)):
        if np.linalg.norm(array[t] - desired_point) < closeness_threshold:
            return t
    return np.nan

## Build mask for target entity (j=3 in paper, index 2 here)

ENTITY_TO_MASK = 2
last_idx = find_last_index_in_interval_where_array_value_is_close_to_desired_point(
        dataset.xs[:, ENTITY_TO_MASK],
        desired_point=np.array([1, 1]),
        starting_index=0,
        ending_index=T,
    )
mask_TJ = np.full((T, J), True)
mask_TJ[last_idx:, ENTITY_TO_MASK] = False
