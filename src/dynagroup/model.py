from dataclasses import dataclass
from typing import Callable


@dataclass
class Model:
    """
    Gives the ingredients necessary to define a hierarchical switching recurrent dynamical model,
    as defined in the NeurIPS submission.

    Note that we are implicitly defining a distribution over observations (y)
    elsewhere.  Namely, this is a linear Gaussian model.  But we haven't made this explict yet because
    Model2a directly observes the x's.
    """

    compute_log_continuous_state_emissions_at_initial_timestep_JAX: Callable
    compute_log_continuous_state_emissions_after_initial_timestep_JAX: Callable
    compute_log_system_transition_probability_matrices_JAX: Callable
    compute_log_entity_transition_probability_matrices_JAX: Callable
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX: Callable
