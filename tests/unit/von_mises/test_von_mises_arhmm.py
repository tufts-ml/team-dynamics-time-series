import numpy as np

from dynagroup.metrics import compute_regime_labeling_accuracy
from dynagroup.util import (
    convert_list_of_regime_id_and_num_timesteps_to_regime_sequence,
)
from dynagroup.von_mises.generate import sample_from_switching_von_mises_AR_with_drift
from dynagroup.von_mises.inference.ar import VonMisesParams
from dynagroup.von_mises.inference.arhmm import run_EM_for_von_mises_arhmm


def test__run_EM_for_von_mises_arhmm():
    """
    Test if we can learn params for the von mises AR-HMM when we don't know the true
    emissions parameters or tpm.
    """
    ###
    # Generate
    ###

    emissions_params_by_regime_true = [
        VonMisesParams(drift=np.pi / 8, kappa=10, ar_coef=0.2),
        VonMisesParams(drift=-np.pi / 8, kappa=100, ar_coef=-0.5),
    ]
    list_of_regime_id_and_num_timesteps = [(0, 100), (1, 100)]
    init_angle = 0.0

    angles = sample_from_switching_von_mises_AR_with_drift(
        emissions_params_by_regime_true,
        list_of_regime_id_and_num_timesteps,
        init_angle,
    )

    ###
    # Inference
    ###

    num_regimes = 2
    self_transition_prob_init = 0.995
    num_EM_iterations = 3

    posterior_summary, emissions_params_by_regime_learned, transitions = run_EM_for_von_mises_arhmm(
        angles, num_regimes, self_transition_prob_init, num_EM_iterations
    )

    ####
    # EVALUATE
    ####

    ACCURACY_THRESHOLD = 0.95

    # TODO: get actual viterbi, not marginal MAP estimates.
    fitted_regime_sequence = np.argmax(posterior_summary.expected_regimes, 1)
    true_regime_sequence = convert_list_of_regime_id_and_num_timesteps_to_regime_sequence(
        list_of_regime_id_and_num_timesteps
    )
    segmentation_accuracy = compute_regime_labeling_accuracy(
        fitted_regime_sequence, true_regime_sequence
    )

    print(f"Segmentation accuracy on von Mises autoregressive HMM was {segmentation_accuracy:.02f}")
    assert segmentation_accuracy >= ACCURACY_THRESHOLD
