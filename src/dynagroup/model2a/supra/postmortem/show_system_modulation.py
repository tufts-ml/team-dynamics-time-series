import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


###
# First gather info
###

entity_idx = 6

### Determine how system regime ordering matchhes to colors
np.argmax(VES_summary.expected_regimes, 1)
# this arbitrary mapping was discovered by (currently manual) postmortem
# TODO: Do this programatically. Problem is I can't preset colors
# because of label swapping.
# green=1, yellow =0, red=2.
system_green_idx = 1
system_yellow_idx = 0
system_red_idx = 2


### get entity regime ordering to match to colors
np.argmax(VEZ_summaries.expected_regimes[:, 6], 1)
# blue =0, red =1, amber =2, green=3,

###  inspect params.. looking at the probs of going into the first state.
np.exp(params_learned.ETP.Ps)[0]


# this arbitrary mapping was discovered by (currently manual) postmortem
entity_look_state = 0

entity_tpms = np.exp(params_learned.ETP.Ps)[entity_idx]

entity_tpm_when_system_is_green = entity_tpms[system_green_idx]
entity_tpm_when_system_is_yellow = entity_tpms[system_yellow_idx]
entity_tpm_when_system_is_red = entity_tpms[system_red_idx]

prob_transition_to_look_state_when_system_is_green = entity_tpm_when_system_is_green[
    :, entity_look_state
]
prob_transition_to_look_state_when_system_is_yellow = entity_tpm_when_system_is_yellow[
    :, entity_look_state
]
prob_transition_to_look_state_when_system_is_red = entity_tpm_when_system_is_red[
    :, entity_look_state
]

###
# Now show conditional probabilities
###

results_matrix = np.array(
    [
        prob_transition_to_look_state_when_system_is_green,
        prob_transition_to_look_state_when_system_is_yellow,
        prob_transition_to_look_state_when_system_is_red,
    ]
).T
# reverse order of rows, so look state is on bottom
results_matrix = results_matrix[::-1]

# Plotting the heatmap
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(results_matrix, annot=True, cmap="Blues", fmt=".2f", ax=ax, cbar=False)

# Customize the plot

ax.set_xticks(np.arange(results_matrix.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(results_matrix.shape[0]) + 0.5, minor=False)

# want a more natural, table-like display
ax.invert_yaxis()
ax.xaxis.tick_top()

ax.set_xticklabels(["Green", "Yellow", "Red"], minor=False)
ax.set_yticklabels(["Green", "Amber", "Red", "Turn North"], minor=False)
ax.set_xlabel("Squad state")
ax.set_ylabel("Soldier state (most recent)")
ax.xaxis.set_label_coords(0.5, 1.15)
# condiitonal prob of turning north
plt.tight_layout()
fig.savefig(save_dir + "system_modulation_of_conditional_probs_of_entity_6_turning_north.pdf")
plt.show()

# ###
# # Or show steady state probabilities
# ###
# # Compute the steady-state probabilities
# ###
# def steady_state_probs_from_tpm(tpm):
#     eigenvalues, eigenvectors = np.linalg.eig(tpm.T)
#     steady_state_index = np.where(np.isclose(eigenvalues, 1))[0][0]
#     steady_state_probabilities = np.real(eigenvectors[:, steady_state_index].T)
#     steady_state_probabilities /= np.sum(steady_state_probabilities)
#     return steady_state_probabilities


# steady_state_probs_from_tpm(entity_tpm_when_system_is_green)
# steady_state_probs_from_tpm(entity_tpm_when_system_is_yellow)
# steady_state_probs_from_tpm(entity_tpm_when_system_is_red)
