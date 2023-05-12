import numpy as np

from dynagroup.hmm_posterior import (
    HMM_Posterior_Summary_NUMPY,
    compute_closed_form_M_step,
)


def test_compute_closed_form_M_step():
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

    posterior_summary = HMM_Posterior_Summary_NUMPY(
        expected_regimes, expected_joints, log_normalizer=None, entropy=None
    )

    tpm = compute_closed_form_M_step(posterior_summary)

    a11_expected = (0.1 + 0.5 + 0.25) / (0.3 + 0.7 + 0.5)
    a12_expected = (0.2 + 0.2 + 0.25) / (0.3 + 0.7 + 0.5)
    a21_expected = (0.3 + 0.2 + 0.25) / (0.7 + 0.3 + 0.5)
    a22_expected = (0.25 + 0.25 + 0.25) / (0.7 + 0.3 + 0.5)

    tpm_expected = np.array([[a11_expected, a12_expected], [a21_expected, a22_expected]])
    assert np.allclose(tpm, tpm_expected, atol=1e-3)
