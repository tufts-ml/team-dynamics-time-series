import numpy as np
import pytest

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summary_NUMPY,
    compute_closed_form_M_step,
    compute_closed_form_M_step_on_posterior_summaries,
    make_hmm_posterior_summaries_from_list,
    make_list_from_hmm_posterior_summaries,
)


@pytest.fixture
def posterior_summary_1():
    # expected_joints: np.array with shape (T-1, K, K)
    #     Gives E[x_{t+1}, x_t | y_{1:T}]; that is, the (t,k,k')-th element gives
    #     the probability distribution over all pairwise options
    #     (x_{t+1}=k', x_{t}=k | y_{1:T}) for t=1,...t-1
    expected_joint_1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    expected_joint_2 = np.array([[0.5, 0.2], [0.2, 0.1]])
    expected_joint_3 = np.array([[0.25, 0.25], [0.25, 0.25]])
    expected_joints = np.array([expected_joint_1, expected_joint_2, expected_joint_3])
    """
    expected_joints looks like this:
    
array([[[0.1 , 0.2 ],
        [0.3 , 0.4 ]],

       [[0.5 , 0.2 ],
        [0.2 , 0.1 ]],

       [[0.25, 0.25],
        [0.25, 0.25]]])
    """
    # expected_regimes: np.array with shape (T,K)
    # Gives E[x_t | y_{1:T}]
    a = 0.6
    expected_regimes = np.array([[0.3, 0.7], [0.7, 0.3], [0.5, 0.5], [a, 1 - a]])
    # expected_regimes has an extra piece of info due to the extra timestep at the end.
    # hence we can fill it in with [a,1-a] for any a in [0,1].

    return HMM_Posterior_Summary_NUMPY(
        expected_regimes, expected_joints, log_normalizer=None, entropy=None
    )


@pytest.fixture
def tpm_expected_from_posterior_summary_1():
    a11_expected = (0.1 + 0.5 + 0.25) / (0.3 + 0.7 + 0.5)
    a12_expected = (0.2 + 0.2 + 0.25) / (0.3 + 0.7 + 0.5)
    a21_expected = (0.3 + 0.2 + 0.25) / (0.7 + 0.3 + 0.5)
    a22_expected = (0.25 + 0.25 + 0.25) / (0.7 + 0.3 + 0.5)

    return np.array([[a11_expected, a12_expected], [a21_expected, a22_expected]])


@pytest.fixture
def posterior_summary_2():
    # expected_joints: np.array with shape (T-1, K, K)
    #     Gives E[x_{t+1}, x_t | y_{1:T}]; that is, the (t,k,k')-th element gives
    #     the probability distribution over all pairwise options
    #     (x_{t+1}=k', x_{t}=k | y_{1:T}) for t=1,...t-1
    expected_joint_1 = np.array([[0.3, 0.2], [0.3, 0.2]])
    expected_joint_2 = np.array([[0.25, 0.25], [0.2, 0.3]])
    expected_joint_3 = np.array([[0.1, 0.5], [0.25, 0.15]])
    expected_joints = np.array([expected_joint_1, expected_joint_2, expected_joint_3])

    # expected_regimes: np.array with shape (T,K)
    # Gives E[x_t | y_{1:T}]
    a = 0.9
    expected_regimes = np.array([[0.5, 0.5], [0.5, 0.5], [0.6, 0.4], [a, 1 - a]])
    # expected_regimes has an extra piece of info due to the extra timestep at the end.
    # hence we can fill it in with [a,1-a] for any a in [0,1].

    return HMM_Posterior_Summary_NUMPY(
        expected_regimes, expected_joints, log_normalizer=None, entropy=None
    )


@pytest.fixture
def tpm_expected_from_posterior_summary_2():
    a11_expected = (0.3 + 0.25 + 0.1) / (0.5 + 0.5 + 0.6)
    a12_expected = (0.2 + 0.25 + 0.5) / (0.5 + 0.5 + 0.6)
    a21_expected = (0.3 + 0.2 + 0.25) / (0.5 + 0.5 + 0.4)
    a22_expected = (0.2 + 0.3 + 0.15) / (0.5 + 0.5 + 0.4)

    return np.array([[a11_expected, a12_expected], [a21_expected, a22_expected]])


@pytest.fixture
def posterior_summaries(posterior_summary_1, posterior_summary_2):
    return make_hmm_posterior_summaries_from_list([posterior_summary_1, posterior_summary_2])


def test_compute_closed_form_M_step(posterior_summary_1, tpm_expected_from_posterior_summary_1):
    tpm_computed_from_posterior_summary_1 = compute_closed_form_M_step(posterior_summary_1)
    assert np.allclose(
        tpm_computed_from_posterior_summary_1, tpm_expected_from_posterior_summary_1, atol=1e-3
    )


def test_conversion_of_hmm_posterior_summaries_to_list_and_back(posterior_summaries):
    list_of_posterior_summaries = make_list_from_hmm_posterior_summaries(posterior_summaries)
    posterior_summaries_recomputed = make_hmm_posterior_summaries_from_list(
        list_of_posterior_summaries
    )
    assert np.allclose(
        posterior_summaries.expected_regimes, posterior_summaries_recomputed.expected_regimes
    )
    assert np.allclose(
        posterior_summaries.expected_joints, posterior_summaries_recomputed.expected_joints
    )
    # TODO: Also test entropies and log normalizers.  Leaving that out for now because they can be None.


def test_compute_closed_form_M_step_on_posterior_summaries(
    posterior_summaries,
    tpm_expected_from_posterior_summary_1,
    tpm_expected_from_posterior_summary_2,
):
    tpms_by_entity = compute_closed_form_M_step_on_posterior_summaries(posterior_summaries)
    assert np.allclose(tpms_by_entity[0], tpm_expected_from_posterior_summary_1, atol=1e-3)
    assert np.allclose(tpms_by_entity[1], tpm_expected_from_posterior_summary_2, atol=1e-3)


def test__compute_closed_form_M_step__with__use_continuous_states__mask(posterior_summary_1):
    # we just use the first two samples...that's ONE transition!
    use_continuous_states = np.array([True, True, False, False])
    tpm_with_mask_expected = np.array([[1 / 3, 2 / 3], [3 / 7, 4 / 7]])

    tpm_with_mask_computed = compute_closed_form_M_step(posterior_summary_1, use_continuous_states)
    assert np.allclose(tpm_with_mask_computed, tpm_with_mask_expected, atol=1e-3)
