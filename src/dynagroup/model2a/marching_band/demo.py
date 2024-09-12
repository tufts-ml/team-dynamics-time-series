import numpy as np
import jax.numpy as jnp
import numpy.random as npr
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import time 

from dynagroup.model2a.marching_band.model_factors import marching_model_JAX
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.hmm_posterior import convert_hmm_posterior_summaries_from_jax_to_numpy
from dynagroup.params import dims_from_params, numpy_params_from_params
from dynagroup.sampler import sample_team_dynamics
from dynagroup.sticky import sample_sticky_transition_matrix
from dynagroup.types import NumpyArray4D
from dynagroup.util import generate_random_covariance_matrix
from dynagroup.util import (
    get_current_datetime_as_string,
    normalize_log_potentials_by_axis,
    segment_list
)
from dynagroup.model2a.marching_band.data.run_sim import system_regimes_gt

from dynagroup.initialize import compute_elbo_from_initialization_results
from dynagroup.vi.M_step_and_ELBO import M_step_toggles_from_strings
from dynagroup.model2a.marching_band.data.run_sim import generate_training_data
from dynagroup.vi.core import SystemTransitionPrior_JAX, run_CAVI_with_JAX
from dynagroup.params import (
    Dims,
    get_dim_of_entity_recurrence_output,
    get_dim_of_system_recurrence_output,
    save_params,
)
from dynagroup.plotting.sampling import plot_sample_with_system_regimes
from dynagroup.model2a.basketball.model import (
    Model_Type,
    get_basketball_model,
    save_model_type,
)
from dynagroup.hmm_posterior import save_hmm_posterior_summary
from dynagroup.io import ensure_dir


"""
Model 2a refers to the fact that here we'll take the x's to be observed.
"""

###
# Configs
###a

# Model specification
n_train_sequences = 10
K = 12
L = 6

# Masking and model adjustments
model_adjustment = None  # Options: None, "one_system_regime", "remove_recurrence"

###
# SPECIFY MODEL
###
model = marching_model_JAX

GLOBAL_MSG = "LAUGH" * n_train_sequences
N = 64
T = 200
total_time = 10650

# Initialization
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.LOCATION

# Inference
n_cavi_iterations = 10
M_step_toggle_for_STP = "gradient_descent"  # "closed_form_tpm"
M_step_toggle_for_ETP = "gradient_descent"
M_step_toggle_for_continuous_state_parameters = "closed_form_gaussian"
M_step_toggle_for_IP = "closed_form_gaussian"
system_covariates = None
num_M_step_iters = 50
alpha_system_prior, kappa_system_prior = 1.0, 10.0 
show_system_states = False 

# Directories
datetime_as_string = get_current_datetime_as_string()
run_description = f"weight_threshold={L}_K={K}_train_CAVI_its_{n_cavi_iterations}_timestamp__{datetime_as_string}"
plots_dir = f"/Users/kgili/team-dynamics-time-series/src/dynagroup/model2a/marching_band/results/plots/{run_description}/"
artifacts_dir = f"/Users/kgili/team-dynamics-time-series/src/dynagroup/model2a/marching_band/results/artifacts/{run_description}/"

###
# I/O
###
ensure_dir(plots_dir)
ensure_dir(artifacts_dir)

# For diagnostics
show_plots_after_learning = True 
T_snippet_for_fit_to_observations = 400


###
# MAKE PRIOR
###
system_transition_prior = SystemTransitionPrior_JAX(alpha_system_prior, kappa_system_prior)

###
# Data splitting and preprocessing
###

gen = generate_training_data(GLOBAL_MSG, N, T, 0)
DATA = gen[0]
example_end_times = gen[1]
cluster_states = gen[2]
true_system_regimes = np.argmax(system_regimes_gt(10, [3333,3394,3730,4824,4889,4969,8919,8977,9036,9093,9168,10314,10376]), axis=1)

###
# MASKING
###
use_continuous_states = None   #QUESTION: WHAT DOES THIS MEAN? 


#### Setup Dims
J = N
D = 2
D_e = get_dim_of_entity_recurrence_output(D, model)
D_s = get_dim_of_system_recurrence_output(D, J, system_covariates, model)
M_e = 0  # for now!
DIMS = Dims(J, K, L, D, D_e, N, D_s, M_e)

###
# INITIALIZATION
###
start_time = time.time() 
print("Running smart initialization.")

### TODO: Make smart initialization better. E.g.
# 1) Run init x times, pick the one with the best ELBO.
# 2) Find a way to do smarter init for the recurrence parameters
# 3) Add prior into the M-step for the system-level tpm (currently it's doing closed form ML).

results_init = smart_initialize_model_2a(
    DIMS,
    DATA,
    example_end_times, 
    model,
    preinitialization_strategy_for_CSP,
    num_em_iterations_for_bottom_half_init,
    num_em_iterations_for_top_half_init,
    seed_for_initialization,
    system_covariates,
    use_continuous_states,
    save_dir=plots_dir,
)
params_init = results_init.params

elbo_init = compute_elbo_from_initialization_results(
    results_init,
    system_transition_prior,
    DATA,
    model,
    example_end_times,
    system_covariates,
)
print(f"ELBO after init: {elbo_init:.02f}")

#QUESTION: HOW WOULD I INPUT A GOOD Q FOR THE MODEL HERE? 

####
# Inference
####

VES_summary, VEZ_summaries, params_learned, elbo_decomposed = run_CAVI_with_JAX(
    jnp.asarray(DATA),
    n_cavi_iterations,
    results_init,
    model,
    example_end_times,
    M_step_toggles_from_strings(
        M_step_toggle_for_STP,
        M_step_toggle_for_ETP,
        M_step_toggle_for_continuous_state_parameters,
        M_step_toggle_for_IP,
    ),
    num_M_step_iters,
    system_transition_prior,
    system_covariates,
    use_continuous_states,
    true_system_regimes,
)


end_time = time.time() 
elapsed_time = end_time - start_time
print(f"Code execution time: {elapsed_time} seconds")

### Save model, learned params, latent state distribution
save_model_type(model, artifacts_dir, basename_prefix=run_description)
save_params(params_learned, artifacts_dir, basename_prefix=run_description)
save_hmm_posterior_summary(VES_summary, "qS", artifacts_dir, basename_prefix=run_description)
save_hmm_posterior_summary(VEZ_summaries, "qZ", artifacts_dir, basename_prefix=run_description)



### CAN THE MODEL EXPLAIN COORDIANTED BEHAVIOR THAT IS CONTEXTUAL ON INDIDIVUAL ENTITY BEHAVIOR? AND INDIVIDUAL ENTITY BEHAVIOR THAT IS CONTEXTUAL ON COORDINATED BEHAVIOR? 
###The soldier dataset does the latter; this dataset does the former. 

if show_plots_after_learning:
    most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1) 
    most_likely_entity_regimes = np.argmax(VEZ_summaries.expected_regimes, axis=2)

    data_np = np.asarray(most_likely_system_regimes)
    split = segment_list(data_np, example_end_times[1:])

    for s, data in enumerate(split[:-1]): 

        #With help from ChatGPT4
        print(f"Plotting the most likely system regimes in time with segments for sequence {s+1}")
        unique_values = np.unique(data)
        letters= ["L", "A", "U", "G", "H", "C"]
        colors = ['#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928','#a6cee3']
        color_map = dict(zip(unique_values, colors))

        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 5))

        for i in range(len(data)):
            value = data[i]
            rect = patches.Rectangle((i, -1), 1, 2, linewidth=0, edgecolor='none', facecolor=color_map[value])
            ax.add_patch(rect)

        # Customize plot
        ax.set_xlim(0, len(data))
        ax.set_ylim(-1, 1)  # Narrow y-axis to fit the line segments
        #ax.set_xticks([0, len(data)])
        ax.set_yticks([])
        ax.set_title(f'System State Segmentation for Sequence {s+1}')

        # Add a legend
        label_map = dict(zip(unique_values, letters))
        handles = [patches.Patch(color=color_map[val], label=f'{label_map[val]}') for val in unique_values]
        ax.legend(handles=handles, loc='upper right')

        # Show plot
        plt.savefig(plots_dir + f"seq{s+1}")
        plt.show()


    # print("Plotting the number of individual players out of bounds in data")

    # num_out_of_bounds = []
    # for i in range(total_time): 
    #     j_list = DATA[i, 0:64, 0]
    #     num_out = 0
    #     for elem in j_list: 
    #         if elem > 1 or elem < 0: 
    #             num_out += 1 
    #     num_out_of_bounds.append(num_out)
    # from IPython import embed; embed()
    # composite_list = [num_out_of_bounds[x:x+1000] for x in range(0, len(num_out_of_bounds),1000)]
    # summed_list = []
    # for sublist in composite_list:
    #     for i in range(len(sublist)):
    #         summed_list[i] += sublist[i]
    
    # ave_out_of_bounds = [element / 10 for element in summed_list]
    # print(f"ave_out_of_bounds = {ave_out_of_bounds}")
    # plt.figure(figsize=(8, 6))
    # plt.plot(ave_out_of_bounds, color="black")
    # plt.xlabel("Time")
    # plt.ylabel("Number of Clumsy Entities")   
    # plt.savefig(plots_dir + "num_clumsy")
    # plt.show()


    # print("Plotting the most likely entity regime for each entity in time")
    # for j in range(DIMS.J):
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(most_likely_entity_regimes[j], color="black")
    #     plt.xlabel("Time")
    #     plt.ylabel("Entity State")   
    #     plt.title(f"Most Likely Entity {j+1} Regime") 
    #     plt.savefig(plots_dir + f"entity_regime_plot_{j+1}")
    #     plt.show()


    # print("Plotting the training data x-values for all j entities")
    # for j in range(DIMS.J):
    #     plt.figure(figsize=(8, 6))
    #     plt.scatter(DATA[0:total_time:20, j, 0], color="black")
    #     plt.xlabel("Time")
    #     plt.ylabel("X-Value")  
    #     plt.title(f"Training Data {j+1}")
    #     plt.savefig(plots_dir + f"entity_regime_plot_{j+1}")
    #     plt.show()


# ###  Plot Fit to Observations     
# if show_plots_after_learning:
#     d = 0
#     for j in range(DIMS.J):
#         for after_learning in [False, True]:
#             predictive_means = compute_next_step_predictive_means(
#                 numpy_params_from_params(params_learned), total_time,
#                 DATA,
#                 convert_hmm_posterior_summaries_from_jax_to_numpy(VEZ_summaries),
#                 after_learning,
#             )
#             print(f"Plotting next-step predictive means for entity {j} and dim {d}")
#             plt.figure(figsize=(8, 6))
#             plt.plot(predictive_means[0:T_snippet_for_fit_to_observations, j, d], color="red")
#             plt.plot(DATA[0:T_snippet_for_fit_to_observations, j, d], color="black")
#             plt.xlabel("Time")
#             plt.ylabel("X-Value")  
#             plt.title(f"Entity {j+1}")
#             plt.savefig(plots_dir + "fit_to_observations")
#             plt.show()








