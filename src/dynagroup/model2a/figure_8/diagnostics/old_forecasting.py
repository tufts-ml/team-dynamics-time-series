import matplotlib.pyplot as plt
import numpy as np

from dynagroup.model2a.figure_8.centers import (
    compute_circle_locations_from_CSP,
    compute_regime_labels_for_up_circle_by_entity,
)
from dynagroup.model2a.figure_8.diagnostics.trajectories import (
    plot_deterministic_trajectories,
)
from dynagroup.model2a.figure_8.generate import (
    log_probs_for_one_step_ahead_entity_transitions__for_figure_8_model,
    log_probs_for_one_step_ahead_system_transitions,
)
from dynagroup.params import AllParameters, dims_from_params
from dynagroup.sampler import sample_team_dynamics


def plot_results_of_old_forecasting_test(
    params: AllParameters,
    T: int,
    title_prefix: str = "generated",
) -> None:
    #### CONFIGS
    seed = 1

    # Initialize the state "x" to be at the right of the "up" circle
    x_init = [1.0, 1.0]

    # Initialize the entity regimes to be at the "up" circle.
    circle_locations = compute_circle_locations_from_CSP(params.CSP)
    fixed_init_entity_regimes = compute_regime_labels_for_up_circle_by_entity(circle_locations)

    for system_regime in [0, 1]:
        print(f"Now showing results when system regime is {system_regime} throughout.")

        DIMS = dims_from_params(params)
        fixed_init_continuous_states = np.tile(np.array(x_init), (DIMS.J, 1))
        fixed_system_regimes = np.array([system_regime] * T)

        sample = sample_team_dynamics(
            params,
            T,
            log_probs_for_one_step_ahead_system_transitions,
            log_probs_for_one_step_ahead_entity_transitions__for_figure_8_model,
            seed=seed,
            fixed_system_regimes=fixed_system_regimes,
            fixed_init_entity_regimes=fixed_init_entity_regimes,
            fixed_init_continuous_states=fixed_init_continuous_states,
        )

        show_plot_type_one = False
        if show_plot_type_one:
            # TODO: I haven't used this in a long time... Just delete it.
            j_fixed = 2
            plt.plot(sample.xs[:, j_fixed, 0], sample.xs[:, j_fixed, 1])
            plt.title(f"Trajectory when system regime = {system_regime} ")
            plt.show()

        else:
            # sample.xs is (T,J,D)
            # want (J,K, num_time_samples, D)

            K = 1
            x_to_show = np.zeros((DIMS.J, K, T, DIMS.D))
            for j in range(DIMS.J):
                for t in range(T):
                    x_to_show[j, 0, t] = sample.xs[t, j]

            plot_deterministic_trajectories(
                x_to_show,
                title_prefix,
                f"under system regime {system_regime}",
                state_entity_regimes_in_subplots=False,
            )
