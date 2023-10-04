from enum import Enum

from dynagroup.model import Model
from dynagroup.model2a.basketball.recurrence import (
    LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
    ZERO_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)
from dynagroup.model2a.gaussian.model_factors import (
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_system_transition_probability_matrices_JAX,
)


###
# Enums
###


class Model_Type(Enum):
    Linear_Recurrence = 1
    No_Recurrence = 2


###
# Models
###

# TODO: Can I set up the entity and system to be generic across fig8 and circles so that we
# call a single function each time?
model_basketball_linear_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)


model_basketball_no_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    ZERO_transform_of_continuous_state_vector_before_premultiplying_by_recurrence_matrix_JAX,
)


def get_basketball_model(model_type_string: str) -> Model:
    return get_basketball_model_from_model_type(Model_Type(model_type_string))


def get_basketball_model_from_model_type(model_type: Model_Type) -> Model:
    if model_type == Model_Type.Linear_Recurrence:
        return model_basketball_linear_recurrence
    elif model_type == Model_Type.No_Recurrence:
        return model_basketball_linear_recurrence
    else:
        raise ValueError("I don't understand the model type.")
