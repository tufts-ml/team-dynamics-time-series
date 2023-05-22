from dynagroup.model import Model
from dynagroup.model2a.basketball.recurrence import (
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
from dynagroup.model2a.gaussian.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)


# TODO: Can I set up the entity and system to be generic across fig8 and circles so that we
# call a single function each time?
model_basketball = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
