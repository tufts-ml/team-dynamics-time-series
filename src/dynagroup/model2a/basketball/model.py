from enum import Enum

from dynagroup.model import Model
from dynagroup.model2a.basketball.recurrence_entity import (
    LINEAR_AND_OOB_AND_COURT_SIDE_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    ZERO_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
)
from dynagroup.model2a.basketball.recurrence_system import (
    ALL_PLAYER_LOCATIONS_system_recurrence_transformation,
    TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation,
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
    No_Recurrence = 1
    Linear_Entity_Recurrence = 2
    Out_of_Bounds_Entity_Recurrence = 3
    Linear_And_Out_Of_Bounds_Entity_Recurrence = 4
    Linear_And_Out_Of_Bounds_And_Court_Side_Entity_Recurrence = 5
    Linear_And_Out_Of_Bounds_Entity_Recurrence__and__Team_Centroid_System_Recurrence = 6
    Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence = 7


###
# Models
###

# TODO: Can I set up the entity and system to be generic across fig8 and circles so that we
# call a single function each time?

model_basketball_no_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    ZERO_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)


model_basketball_linear_entity_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)


model_basketball_out_of_bounds_entity_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)

model_basketball_linear_and_out_of_bounds_entity_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)


model_basketball_linear_and_out_of_bounds_and_court_side_entity_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_AND_OOB_AND_COURT_SIDE_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX=None,
)

model_basketball_linear_and_out_of_bounds_entity_recurrence__and__team_centroid_system_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    TEAM_CENTROID_X_DISTANCE_system_recurrence_transformation,
)

model_basketball_linear_and_out_of_bounds_entity_recurrence__and__all_player_locations_system_recurrence = Model(
    compute_log_initial_continuous_state_emissions_JAX,
    compute_log_continuous_state_emissions_after_initial_timestep_JAX,
    compute_log_system_transition_probability_matrices_JAX,
    compute_log_entity_transition_probability_matrices_JAX,
    LINEAR_AND_OOB_RECURRENCE_transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX,
    ALL_PLAYER_LOCATIONS_system_recurrence_transformation,
)


def get_basketball_model(model_type_string: str) -> Model:
    return get_basketball_model_from_model_type(Model_Type(model_type_string))


def get_basketball_model_from_model_type(model_type: Model_Type) -> Model:
    if model_type == Model_Type.No_Recurrence:
        return model_basketball_no_recurrence
    elif model_type == Model_Type.Linear_Entity_Recurrence:
        return model_basketball_linear_entity_recurrence
    elif model_type == Model_Type.Out_of_Bounds_Entity_Recurrence:
        return model_basketball_out_of_bounds_entity_recurrence
    elif model_type == Model_Type.Linear_And_Out_Of_Bounds_Entity_Recurrence:
        return model_basketball_linear_and_out_of_bounds_entity_recurrence
    elif model_type == Model_Type.Linear_And_Out_Of_Bounds_And_Court_Side_Entity_Recurrence:
        return model_basketball_linear_and_out_of_bounds_and_court_side_entity_recurrence
    elif model_type == Model_Type.Linear_And_Out_Of_Bounds_Entity_Recurrence__and__Team_Centroid_System_Recurrence:
        return model_basketball_linear_and_out_of_bounds_entity_recurrence__and__team_centroid_system_recurrence
    elif (
        model_type == Model_Type.Linear_And_Out_Of_Bounds_Entity_Recurrence__and__All_Player_Locations_System_Recurrence
    ):
        return model_basketball_linear_and_out_of_bounds_entity_recurrence__and__all_player_locations_system_recurrence
    else:
        raise ValueError("I don't understand the model type.")
