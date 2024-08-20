from dynagroup.model import Model
from dynagroup.model2a.figure8.recurrence import (
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
)
from dynagroup.model2a.gaussian.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)

marching_model_JAX = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)



