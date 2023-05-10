import warnings
from dataclasses import dataclass
from typing import Union

import jax.numpy as jnp
import jax_dataclasses as jdc
import numpy as np
from jax import vmap

from dynagroup.covariances import (
    cholesky_nzvals_from_covariance_JAX,
    covariance_from_cholesky_nzvals_JAX,
)
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
from dynagroup.util import (
    normalize_log_potentials_by_axis_JAX,
    tpm_from_unconstrained_tpm,
    unconstrained_tpm_from_tpm,
)


###
# Parameters
###


@dataclass
class SystemTransitionParameters:
    """
    Attributes:
        Gammas: has shape (J, L, K)
            Set this to zero if there is no entity-to-system feedback.
        Upsilon: has shape (L, M_s):
            Weights for covariates on system transitions
        Pi: has shape (L, L).
            Log of a LxL transition probability matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        M_s: dimensionality of system-level covariates

    Remarks:
        It may seem weird to represent Pi here, even though exp(Pi) is the transition probability matrix,
        and even though we do numerical optimization on a non-overparametrized representation of Pi.  However,
        Pi, not exp(Pi), is the form that enters into the linear predictor function of the Categorical GLM. See
        the state space modeling notes.
    """

    Gammas: NumpyArray3D
    Upsilon: NumpyArray2D
    Pi: NumpyArray2D


@jdc.pytree_dataclass
class SystemTransitionParameters_JAX:
    """
    Attributes:
        Gammas: has shape (J, L, K)
            Set this to zero if there is no entity-to-system feedback.
        Upsilon: has shape (L, M_s):
            Weights for covariates on system transitions
        Pi: has shape (L, L).
            Log of a LxL transition probability matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        M_s: dimensionality of system-level covariates

    Remarks:
        It may seem weird to represent Pi here, even though exp(Pi) is the transition probability matrix,
        and even though we do numerical optimization on a non-overparametrized representation of Pi.  However,
        Pi, not exp(Pi), is the form that enters into the linear predictor function of the Categorical GLM. See
        the state space modeling notes.
    """

    Gammas: JaxNumpyArray3D
    Upsilon: JaxNumpyArray2D
    Pi: JaxNumpyArray2D


@dataclass
class EntityTransitionParameters_MetaSwitch:
    """
    Attributes:
        Psis : has shape (J, L, K, D_t)
            Each Psis[j] gives recurrence weights from the continuous states,
            after we've transformed the continuous states from dim D to dim D_t.
            The L dimension is to switch between different matrices with shape (K,D_t).
        Omegas : has shape (J, L, K, M_e)
            Each Omegas[j] gives covariance weights.
            The L dimension is to switch between different KxL matrices
        Ps : has shape (J, L, K, K)
            Each Ps[j,l] is the log of a KxK transition probability matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t: dimensionality of latent continuous state, x, after applying transformation f.
        M_e: dimensionality of entity-level covariates
    """

    Psis: NumpyArray4D
    Omegas: NumpyArray4D
    Ps: NumpyArray4D


@jdc.pytree_dataclass
class EntityTransitionParameters_MetaSwitch_JAX:
    """
    Attributes:
        Psis : has shape (J, L, K, D_t)
            Each Psis[j] gives recurrence weights from the continuous states,
            after we've transformed the continuous states from dim D to dim D_t.
            The L dimension is to switch between different matrices with shape (K,D_t).
        Omegas : has shape (J, L, K, M_e)
            Each Omegas[j] gives covariance weights.
            The L dimension is to switch between different KxL matrices
        Ps : has shape (J, L, K, K)
            Each Ps[j,l] is the log of a KxK transition probability matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t: dimensionality of latent continuous state, x, after applying transformation f.
        M_e: dimensionality of entity-level covariates
    """

    Psis: JaxNumpyArray4D
    Omegas: JaxNumpyArray4D
    Ps: JaxNumpyArray4D


@dataclass
class EntityTransitionParameters_SystemBias:
    """
    Attributes:
        Psis : has shape (J, K, D_t)
            Each Psis[j] gives recurrence weights from the continuous states,
            after we've transformed the continuous states from dim D to dim D_t.
        Omegas : has shape (J, K, M_e)
            Each Omegas[j] gives covariance weights.
        Ps : has shape (J, K, K)
            Each Ps[j] is the log of a KxK transition probability matrix
        Xis : has shape (J, K, L)

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t: dimensionality of latent continuous state, x, after applying transformation f.
        M_e: dimensionality of entity-level covariates
    """

    Psis: NumpyArray3D
    Omegas: NumpyArray3D
    Ps: NumpyArray3D
    Xis: NumpyArray3D


@jdc.pytree_dataclass
class EntityTransitionParameters_SystemBias_JAX:
    """

    Attributes:
        Psis : has shape (J, K, D_t)
            Each Psis[j] gives recurrence weights from the continuous states,
            after we've transformed the continuous states from dim D to dim D_t.
        Omegas : has shape (J, K, M_e)
            Each Omegas[j] gives covariance weights.
        Ps : has shape (J, K, K)
            Each Ps[j] is the log of a KxK transition probability matrix
        Xis : has shape (J, K, L)

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t: dimensionality of latent continuous state, x, after applying transformation f.
        M_e: dimensionality of entity-level covariates
    """

    Psis: JaxNumpyArray3D
    Omegas: JaxNumpyArray3D
    Ps: JaxNumpyArray3D
    Xis: JaxNumpyArray3D


EntityTransitionParameters = Union[
    EntityTransitionParameters_SystemBias, EntityTransitionParameters_MetaSwitch
]

EntityTransitionParameters_JAX = Union[
    EntityTransitionParameters_SystemBias_JAX, EntityTransitionParameters_MetaSwitch_JAX
]


@dataclass
class ContinuousStateParameters_Gaussian:
    """
    Attributes:
        As : has shape (J, K, D, D)
        bs : has shape (J, K, D)
        Qs : has shape (J, K, D,D)
            Each Qs[j] is a covariance matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
    """

    As: NumpyArray4D
    bs: NumpyArray3D
    Qs: NumpyArray4D


@jdc.pytree_dataclass
class ContinuousStateParameters_Gaussian_JAX:
    """
    Attributes:
        As : has shape (J, K, D, D)
        bs : has shape (J, K, D)
        Qs : has shape (J, K, D,D)
            Each Qs[j] is a covariance matrix

    Notation:
        J: number of entities
        K: number of entity-level regimes
        D: dimensionality of latent continuous state, x
    """

    As: JaxNumpyArray4D
    bs: JaxNumpyArray3D
    Qs: JaxNumpyArray4D


@dataclass
class ContinuousStateParameters_VonMises:
    """
    Attributes:
        ar_coefs : has shape (J, K)
        drifts : has shape (J, K)
        kappas : has shape (J, K)

    Notation:
        J: number of entities
        K: number of entity-level regimes
    """

    ar_coefs: JaxNumpyArray2D
    drifts: JaxNumpyArray2D
    kappas: JaxNumpyArray2D


@jdc.pytree_dataclass
class ContinuousStateParameters_VonMises_JAX:
    """
    Attributes:
        ar_coefs : has shape (J, K)
        drifts : has shape (J, K)
        kappas : has shape (J, K)

    Notation:
        J: number of entities
        K: number of entity-level regimes
    """

    ar_coefs: JaxNumpyArray2D
    drifts: JaxNumpyArray2D
    kappas: JaxNumpyArray2D


ContinuousStateParameters = Union[
    ContinuousStateParameters_Gaussian, ContinuousStateParameters_VonMises
]
ContinuousStateParameters_JAX = Union[
    ContinuousStateParameters_Gaussian_JAX, ContinuousStateParameters_VonMises_JAX
]


@dataclass
class EmissionsParameters:
    """
    Attributes:
        Cs : has shape (J, N, D)
        ds : has shape (J, N)
        Rs : has shape (J, N, N)
            Each Rs[j] is a covariance matrix

    Remarks:
        We assume emissions parameters are tied across the K discrete states.
        This assumption is typical in the literature - e.g. see Linderman
        or Emily Fox.

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    Cs: NumpyArray3D
    ds: NumpyArray2D
    Rs: NumpyArray3D


@jdc.pytree_dataclass
class EmissionsParameters_JAX:
    """
    Attributes:
        Cs : has shape (J, N, D)
        ds : has shape (J, N)
        Rs : has shape (J, N, N)
            Each Rs[j] is a covariance matrix

    Remarks:
        We assume emissions parameters are tied across the K discrete states.
        This assumption is typical in the literature - e.g. see Linderman
        or Emily Fox.

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    Cs: JaxNumpyArray3D
    ds: JaxNumpyArray2D
    Rs: JaxNumpyArray3D


@dataclass
class InitializationParameters_Gaussian:
    """
    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        mu_0s : has shape (J,K,D)
            Mean of MVN density on initial continuous state x0
        Sigma_0s : has shape (J,K,D,D)
            Covariance of MVN density on initial continuous state x0
    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    pi_system: NumpyArray1D
    pi_entities: NumpyArray2D
    mu_0s: NumpyArray3D
    Sigma_0s: NumpyArray4D


@jdc.pytree_dataclass
class InitializationParameters_Gaussian_JAX:
    """
    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        mu_0s : has shape (J,K,D)
            Mean of MVN density on initial continuous state x0
        Sigma_0s : has shape (J,K,D,D)
            Covariance of MVN density on initial continuous state x0
    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    pi_system: JaxNumpyArray1D
    pi_entities: JaxNumpyArray2D
    mu_0s: JaxNumpyArray3D
    Sigma_0s: JaxNumpyArray4D


@dataclass
class InitializationParameters_VonMises:
    """
    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        mu_0s : has shape (J,K,D)
            Mean of MVN density on initial continuous state x0
        Sigma_0s : has shape (J,K,D,D)
            Covariance of MVN density on initial continuous state x0
    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    pi_system: NumpyArray1D
    pi_entities: NumpyArray2D
    locs: JaxNumpyArray2D
    kappas: JaxNumpyArray2D


@jdc.pytree_dataclass
class InitializationParameters_VonMises_JAX:
    """
    Attributes:
        pi_system : has shape (L,)
            Lives on the simplex
        pi_entities : has shape (J, K)
            Each pi_entities[j] lives on the simplex.
        locs : has shape (J, K)
            Location parameter for VonMises density on initial continuous state x0
        kappas : has shape (J, K)
            Concentration parameter for VonMises density on initial continuous state x0

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        N : dimensionality of observation, y
    """

    pi_system: JaxNumpyArray1D
    pi_entities: JaxNumpyArray2D
    locs: JaxNumpyArray2D
    kappas: JaxNumpyArray2D


@dataclass
class ContinuousStateParameters_VonMises:
    """
    Attributes:
        locs : has shape (J, K)
            Location parameter for VonMises density on initial continuous state x0
        kappas : has shape (J, K)
            Concentration parameter for VonMises density on initial continuous state x0

    Notation:
        J: number of entities
        K: number of entity-level regimes
    """

    locs: JaxNumpyArray2D
    kappas: JaxNumpyArray2D


InitializationParameters = Union[
    InitializationParameters_Gaussian,
    InitializationParameters_VonMises,
]
InitializationParameters_JAX = Union[
    InitializationParameters_Gaussian_JAX,
    InitializationParameters_VonMises_JAX,
]


@dataclass
class AllParameters:
    STP: SystemTransitionParameters
    ETP: EntityTransitionParameters
    CSP: ContinuousStateParameters
    EP: EmissionsParameters
    IP: InitializationParameters


@jdc.pytree_dataclass
class AllParameters_JAX:
    STP: SystemTransitionParameters_JAX
    ETP: EntityTransitionParameters_JAX
    CSP: ContinuousStateParameters_JAX
    EP: EmissionsParameters_JAX
    IP: InitializationParameters_JAX


###
# Dims
###


@dataclass
class Dims:
    """
    Attributes:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t : dimensionality of transformed latent continuous state, x_tilde, before application of recurrence matrix
        N : dimensionality of observation, y
        M_s: dimensionality of system-level covariates
        M_e: dimensionality of entity-level covariates
    """

    J: int
    K: int
    L: int
    D: int
    D_t: int
    N: int
    M_s: int
    M_e: int


def dims_from_params(all_params: AllParameters) -> Dims:
    return Dims(
        J=np.shape(all_params.STP.Gammas)[0],
        K=np.shape(all_params.STP.Gammas)[2],
        L=np.shape(all_params.STP.Gammas)[1],
        D=np.shape(all_params.EP.Cs)[2],
        D_t=np.shape(all_params.ETP.Psis)[3],
        N=np.shape(all_params.EP.Cs)[1],
        M_s=np.shape(all_params.STP.Upsilon)[1],
        M_e=np.shape(all_params.ETP.Omegas)[3],
    )


###
# Conversions: Jax -> Numpy
###


def numpyify_param_group(param_group_instance, New_Param_Group_Class):
    """
    Takes a parameter group, e.g. `InitializationParameters`, and constructs
    a corresponding jax parameter group,  e.g. `InitializationParameters_JAX`,
    which has the same attributes but as jnp.arrays instead of np.arrays.

    Example:
        numpyify_param_group(all_params.IP, InitializationParameters_JAX)
    """
    dict_of_numpy_arrays = {}
    for attr, value in vars(param_group_instance).items():
        dict_of_numpy_arrays[attr] = np.asarray(value)

    return New_Param_Group_Class(**dict_of_numpy_arrays)


def jaxify_param_group(param_group_instance, New_Param_Group_Class):
    """
    Takes a parameter group, e.g. `InitializationParameters`, and constructs
    a corresponding jax parameter group,  e.g. `InitializationParameters_JAX`,
    which has the same attributes but as jnp.arrays instead of np.arrays.

    Example:
        jaxify_param_group(all_params.IP, InitializationParameters_JAX)
    """
    for attr, value in vars(param_group_instance).items():
        setattr(param_group_instance, attr, jnp.asarray(value))

    return New_Param_Group_Class(**vars(param_group_instance))


def jax_params_from_params(all_params: AllParameters) -> AllParameters_JAX:
    # TODO: I shouldn't assume the type is EntityTransitionParameters_MetaSwitch
    # rather than EntityTransitionParameters_SystemBias.  Is there a way to read in the
    # type programatically?

    # I wanted to do  isinstance(all_params, AllParameters_JAX)
    # but that evaluates to false, even when we have..
    # In [16]: type(all_params)
    # Out[16]: dynagroup.params.AllParameters_JAX
    # Perhaps this is something about jdc package that differs from dataclasses?

    if "AllParameters_JAX" in str(type(all_params)):
        warnings.warn("Parameters ALREADY have Jax typing.")
        return all_params

    return AllParameters_JAX(
        jaxify_param_group(all_params.STP, SystemTransitionParameters_JAX),
        jaxify_param_group(all_params.ETP, EntityTransitionParameters_MetaSwitch_JAX),
        jaxify_param_group(all_params.CSP, ContinuousStateParameters_JAX),
        jaxify_param_group(all_params.EP, EmissionsParameters_JAX),
        jaxify_param_group(all_params.IP, InitializationParameters_JAX),
    )


def numpy_params_from_params(all_params: AllParameters_JAX) -> AllParameters:
    # TODO: I shouldn't assume the type is EntityTransitionParameters_MetaSwitch
    # rather than EntityTransitionParameters_SystemBias.  Is there a way to read in the
    # type programatically?
    return AllParameters_JAX(
        numpyify_param_group(all_params.STP, SystemTransitionParameters),
        numpyify_param_group(all_params.ETP, EntityTransitionParameters_MetaSwitch),
        numpyify_param_group(all_params.CSP, ContinuousStateParameters),
        numpyify_param_group(all_params.EP, EmissionsParameters),
        numpyify_param_group(all_params.IP, InitializationParameters),
    )


###
# Conversions for M-step: Unconstrained representations for transition probability matrices
###


@jdc.pytree_dataclass
class SystemTransitionParameters_WithUnconstrainedTPMs_JAX:
    """
    Attributes:
        Gammas: has shape (J, L, K)
            Set this to zero if there is no entity-to-system feedback.
        Upsilon: has shape (L, M_s):
            Weights for covariates on system transitions
        PiTilde_Unconstrained: has shape (L, L-1).
            Whereas Pi is the log of a LxL transition probability matrix
            (so PiTilde := jnp.exp(Pi) is a LxL transition probability matrix whose rows
            are constrained to live on the simplex with L entries), we here represent
            each row with only L-1 UNCONSTRAINED entries.  This makes for easier optimization;
            we don't have to worry about respecting parameter constraints
            (as if we worked with jnp.exp(Pi)),  nor do we have to worry about overparametrized representations
            (as if we worked with Pi).

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        M_s: dimensionality of system-level covariates
    """

    Gammas: JaxNumpyArray3D
    Upsilon: JaxNumpyArray2D
    PiTilde_Unconstrained: JaxNumpyArray2D


def STP_with_unconstrained_tpms_from_ordinary_STP(
    STP: SystemTransitionParameters_JAX,
) -> SystemTransitionParameters_WithUnconstrainedTPMs_JAX:
    PiTilde_Unconstrained = unconstrained_tpm_from_tpm(jnp.exp(STP.Pi))
    return SystemTransitionParameters_WithUnconstrainedTPMs_JAX(
        STP.Gammas, STP.Upsilon, PiTilde_Unconstrained
    )


def ordinary_STP_from_STP_with_unconstrained_tpms(
    STP_WUC: SystemTransitionParameters_WithUnconstrainedTPMs_JAX,
) -> SystemTransitionParameters_JAX:
    PiTilde = tpm_from_unconstrained_tpm(STP_WUC.PiTilde_Unconstrained)
    return SystemTransitionParameters_JAX(STP_WUC.Gammas, STP_WUC.Upsilon, jnp.log(PiTilde))


@jdc.pytree_dataclass
class EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX:
    """
    Attributes:
        Psis : has shape (J, L, K, D_t)
            Each Psis[j] gives recurrence weights from the continuous states,
            after we've transformed the continuous states from dim D to dim D_t.
            The L dimension is to switch between different matrices with shape (K,D_t).
        Omegas : has shape (J, L, K, M_e)
            Each Omegas[j] gives covariance weights.
            The L dimension is to switch between different KxL matrices
        PTildes_Unconstrained : has shape (J, L, K, K-1)
            Whereas each Ps[j,l] is the log of a LxL transition probability matrix
            (so if PTildes := jnp.exp(Ps), then each PTildes[j,l] is a KxK transition probability matrix whose rows
            are constrained to live on the simplex with K entries), we here represent
            each row with only K-1 UNCONSTRAINED entries.  This makes for easier optimization;
            we don't have to worry about respecting parameter constraints
            (as if we worked with jnp.exp(Ps)),  nor do we have to worry about overparametrized representations
            (as if we worked with Ps).

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
        D_t: dimensionality of latent continuous state, x, after applying transformation f.
        M_e: dimensionality of entity-level covariates
    """

    Psis: JaxNumpyArray4D
    Omegas: JaxNumpyArray4D
    PTildes_Unconstrained: JaxNumpyArray4D


def ETP_MetaSwitch_with_unconstrained_tpms_from_ordinary_ETP_MetaSwitch(
    ETP: EntityTransitionParameters_MetaSwitch_JAX,
) -> EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX:
    PTildes_Unconstrained = unconstrained_tpm_from_tpm(jnp.exp(ETP.Ps))
    return EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX(
        ETP.Psis, ETP.Omegas, PTildes_Unconstrained
    )


def ordinary_ETP_MetaSwitch_from_ETP_MetaSwitch_with_unconstrained_tpms(
    ETP_WUC: EntityTransitionParameters_MetaSwitch_WithUnconstrainedTPMs_JAX,
) -> EntityTransitionParameters_MetaSwitch_JAX:
    PTildes = tpm_from_unconstrained_tpm(ETP_WUC.PTildes_Unconstrained)
    return EntityTransitionParameters_MetaSwitch_JAX(ETP_WUC.Psis, ETP_WUC.Omegas, jnp.log(PTildes))


###
# Conversions for M-step: Unconstrained representations for covariance matrices
###

# TODO: Use tensorflow probability to make these conversions, like we do for simplex-valued parameters.


# @jit
def cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX(batched):
    return vmap(vmap(cholesky_nzvals_from_covariance_JAX))(batched)


# @jit
def covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX(batched):
    return vmap(vmap(covariance_from_cholesky_nzvals_JAX))(batched)


@jdc.pytree_dataclass
class ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX:
    """
    Attributes:
        As : has shape (J, K, D, D)
        bs : has shape (J, K, D)
        Cholesky_nzvals : has shape (J, K, R)
            Each (j,k)-th entry gives R values which gives the nzvals
            of a lower tringular Cholesky factor from which one can
            obtain the covariance matrix.

    Notation:
        J: number of entities
        K: number of entity-level regimes
        L: number of system-level regimes
        D: dimensionality of latent continuous state, x
    """

    As: JaxNumpyArray4D
    bs: JaxNumpyArray3D
    cholesky_nzvals: JaxNumpyArray4D


def CSP_Gaussian_with_unconstrained_covariances_from_ordinary_CSP_Gaussian(
    CSP: ContinuousStateParameters_Gaussian_JAX,
) -> ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX:
    cholesky_nzvals = cholesky_nzvals_from_covariances_with_two_mapping_axes_JAX(CSP.Qs)
    return ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX(
        CSP.As, CSP.bs, cholesky_nzvals
    )


def ordinary_CSP_Gaussian_from_CSP_Gaussian_with_unconstrained_covariances(
    CSP_WUC: ContinuousStateParameters_Gaussian_WithUnconstrainedCovariances_JAX,
) -> ContinuousStateParameters_Gaussian_JAX:
    Qs = covariance_from_cholesky_nzvals_with_two_mapping_axes_JAX(CSP_WUC.cholesky_nzvals)
    return ContinuousStateParameters_Gaussian_JAX(CSP_WUC.As, CSP_WUC.bs, Qs)


###
# Normalizations
###

# TODO: I think `normalize_log_tpms_within_parameter_group` has been rendered unnecessary now that we're using tensorflow probability
# to convert to simplex and back


def normalize_log_tpms_within_parameter_group(
    param_group, name_of_param_to_log_normalize: str, axis: int
):
    """
    Takes a parameter group, e.g. `SystemTransitionParameters`, and normalizes (on a log scale)
    a parameter within that gorup.

    Example:
        STP_new = normalize_log_tpms_within_parameter_group(STP, "Pi", 1)
    """
    object_to_log_normalize = param_group.__dict__[name_of_param_to_log_normalize]
    log_normalized_object = normalize_log_potentials_by_axis_JAX(object_to_log_normalize, axis)
    dict_for_new_param_group = param_group.__dict__
    dict_for_new_param_group[name_of_param_to_log_normalize] = log_normalized_object
    return type(param_group)(**dict_for_new_param_group)
