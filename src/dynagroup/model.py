import os
from dataclasses import dataclass
from typing import Callable, Optional


@dataclass
class Model:
    """
    Gives the ingredients necessary to define a hierarchical switching recurrent dynamical model,
    as defined in the NeurIPS submission.

    Note that we are implicitly defining a distribution over observations (y)
    elsewhere.  Namely, this is a linear Gaussian model.  But we haven't made this explict yet because
    Model2a directly observes the x's.
    """

    compute_log_initial_continuous_state_emissions_JAX: Callable
    compute_log_continuous_state_emissions_after_initial_timestep_JAX: Callable
    compute_log_system_transition_probability_matrices_JAX: Callable
    compute_log_entity_transition_probability_matrices_JAX: Callable
    transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX: Callable
    transform_of_flattened_continuous_state_vectors_before_premultiplying_by_system_recurrence_matrix_JAX: Optional[
        Callable
    ] = None


def save_model_string(model: Model, save_dir: str, basename_postfix: str = ""):
    save_filepath = os.path.join(save_dir, f"model_string_{basename_postfix}.txt")
    fields = [(attribute, value) for attribute, value in model.__dict__.items()]
    with open(save_filepath, "w") as file:
        for field in fields:
            file.write(f"{field[0]}: {field[1]} \n\n")
