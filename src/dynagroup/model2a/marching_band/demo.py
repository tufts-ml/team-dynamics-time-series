import numpy as np
import jax.numpy as jnp
import numpy.random as npr
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
import time 
import os

from dynagroup.hmm_posterior import convert_hmm_posterior_summaries_from_jax_to_numpy

from dynagroup.model2a.marching_band.model_factors import marching_model_JAX
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.metrics import get_aligned_estimate
from dynagroup.util import (
    get_current_datetime_as_string,
    find_indices
)
from sklearn.cluster import KMeans
from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.model2a.marching_band.data.run_sim import system_regimes_gt
from dynagroup.params import numpy_params_from_params
from dynagroup.model2a.marching_band.metrics import compute_next_step_predictive_means
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
from dynagroup.model2a.marching_band.model import (
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
K = 4
L = 6

###
# SPECIFY MODEL
###
model = marching_model_JAX
model_adjustment = None  # Options: None, "one_system_regime", "remove_recurrence"

GLOBAL_MSG = "LAUGH" * n_train_sequences
J = 64
T = 200
total_time = 10300


# Initialization
seed_for_initialization = 1
num_em_iterations_for_bottom_half_init = 1
num_em_iterations_for_top_half_init = 1
preinitialization_strategy_for_CSP = PreInitialization_Strategy_For_CSP.LOCATION 

# Inference
seed = 121 #Need to change in Vi.Core if you want reproducibility over entire training 
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
run_description = f"seed_{seed}_timestamp__{datetime_as_string}_none"
home_dir = os.path.expanduser("~")
plots_dir = f"{home_dir}/team-dynamics-time-series/src/dynagroup/model2a/marching_band/results/plots/{run_description}/"
artifacts_dir = f"{home_dir}/team-dynamics-time-series/src/dynagroup/model2a/marching_band/results/artifacts/{run_description}/"
frames_dir = f"{home_dir}/team-dynamics-time-series/src/dynagroup/model2a/marching_band/results/frames/"

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

gen = generate_training_data(GLOBAL_MSG, J, T, 0)
DATA = gen[0]
example_end_times = gen[1]
cluster_states = gen[2]
true_system_regimes = np.argmax(system_regimes_gt(10,  [1227, 2840, 6128, 7392, 9553, 9680]), axis=1)
###
# MASKING
###
use_continuous_states = None  



#### Setup Dims
N = 0
D = 2
D_e = get_dim_of_entity_recurrence_output(D, model)
D_s = get_dim_of_system_recurrence_output(D, J, system_covariates, model)
M_e = 0 
DIMS = Dims(J, K, L, D, D_e, N, D_s, M_e)

###
# MODEL ADJUSTMENTS
###
if model_adjustment == "one_system_regime":
    DIMS.L = 1
elif model_adjustment == "remove_recurrence":
    model.transform_of_continuous_state_vector_before_premultiplying_by_entity_recurrence_matrix_JAX = (
        lambda x_vec: np.zeros(DIMS.D_e)  
    )


###
# INITIALIZATION
###
start_time = time.time() 
print("Running smart initialization.")


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


####
# Inference
####

VES_summary, VEZ_summaries, params_learned, elbo_decomposed, classification_accuracy = run_CAVI_with_JAX(
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

####
# Plotting and Model Validation for Data 
####

def plot_ca(classification_accuracy): 
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.plot(classification_accuracy)
    plt.xlabel("Number of CAVI Iterations")
    plt.ylabel("Accuracy")
    plt.title("Classification Accuracy")
    plt.savefig(plots_dir + f"accuracy")
    plt.show()

def plot_system_segments(system_data): 
    letters = ["L", "A", "U", "G", "H", "C"]
    ground_truth = true_system_regimes
    colors=['#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928','#a6cee3']

    accuracy = compute_regime_labeling_accuracy(system_data, ground_truth)
    print(accuracy)
    unique_values = np.unique(system_data)
    
    color_map = dict(zip(unique_values, colors[:len(unique_values)]))
    letter_map = dict(zip(unique_values, letters[:len(unique_values)]))
    
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, value in enumerate(system_data):
        rect = patches.Rectangle((i, -1), 1, 2, linewidth=0, edgecolor='none', facecolor=color_map[value])
        ax.add_patch(rect)
        ax.text(i + 0.5, 0, letter_map[value], ha='center', va='center', fontsize=12, color='black')

    ax.set_xlim(0, len(system_data))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('HSRDM')
    handles = [patches.Patch(color=color_map[val], label=f'{letter_map[val]}') for val in unique_values]
    ax.legend(handles=handles, loc='upper right')
    plt.savefig(plots_dir + f"system_states")
    plt.show()
    
def check_cluster_similarity(system_data, true_data, target = 5):
    index_list1 = find_indices(system_data, target)
    index_list2 = find_indices(true_data, target)
    similarity_score = len(set(index_list1).intersection(index_list2))
    return similarity_score

def plot_k_means_entities(entity_data):
    letters = ["L", "A", "U", "G", "H", "C"]
    ground_truth = true_system_regimes
    k = 6
    T = len(entity_data)
    one_hot_encoded = np.eye(k)[entity_data]
    reshaped_data = one_hot_encoded.reshape(T, -1)
    kmeans = KMeans(n_clusters=6, random_state=0)
    cluster_labels = kmeans.fit_predict(reshaped_data)
    colors=['#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928','#a6cee3']

    accuracy = compute_regime_labeling_accuracy(cluster_labels, ground_truth)
    print(accuracy)
    aligned_estimate = get_aligned_estimate(cluster_labels, ground_truth)
    unique_values = np.unique(aligned_estimate)
    
    color_map = dict(zip(unique_values, colors[:len(unique_values)]))
    letter_map = dict(zip(unique_values, letters[:len(unique_values)]))
    
    fig, ax = plt.subplots(figsize=(10, 2))
    for i, value in enumerate(aligned_estimate):
        # Draw the rectangle segment for each predicted cluster
        rect = patches.Rectangle((i, -1), 1, 2, linewidth=0, edgecolor='none', facecolor=color_map[value])
        ax.add_patch(rect)
        # Add text to the middle of the rectangle for each segment
        ax.text(i + 0.5, 0, letter_map[value], ha='center', va='center', fontsize=12, color='black')

    # Customize plot
    ax.set_xlim(0, len(aligned_estimate))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_title('HSRDM No System')
    handles = [patches.Patch(color=color_map[val], label=f'{letter_map[val]}') for val in unique_values]
    ax.legend(handles=handles, loc='upper right')
    plt.show()


if show_plots_after_learning:
    most_likely_system_regimes = np.argmax(VES_summary.expected_regimes, axis=1) 
    most_likely_entity_regimes = np.argmax(VEZ_summaries.expected_regimes, axis=2)

    # Plot system sequence trained model plots 
    system_aligned_sequence = get_aligned_estimate(most_likely_system_regimes, true_system_regimes)
    system_raw = np.asarray(system_aligned_sequence)
    plot_system_segments(system_raw)

    #Plot classification accuracy throughout training
    plot_ca(classification_accuracy)

    plot_k_means_entities(most_likely_entity_regimes)

    #Check cluster state accuracy 
    cluster_metric = check_cluster_similarity(system_raw, true_system_regimes)
    num_identified_cluster_states = np.count_nonzero(system_aligned_sequence == 5)
    print(f"The cluster true positive classification accuracy is {cluster_metric/300}. The total number of identified cluster states is {num_identified_cluster_states}.")




            







