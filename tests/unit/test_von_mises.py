import numpy as np
from scipy.stats import vonmises

from dynagroup.von_mises.generate import (
    sample_from_von_mises_AR_with_drift,
    sample_from_von_mises_random_walk,
    sample_from_von_mises_random_walk_with_drift,
)
from dynagroup.von_mises.inference import VonMisesModelType, estimate_von_mises_params
from dynagroup.von_mises.util import (
    angles_from_points,
    points_from_angles,
    two_angles_are_close,
)


def test__points_from_angles__then__angles_from_points():
    angles = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
    points = points_from_angles(angles)
    angles_reconstructed = angles_from_points(points)
    assert np.allclose(angles, angles_reconstructed)


def test_that__estimate_von_mises_params__gives_approximately_correct_parameters_for_iid_von_mises_model():
    T = 1000
    print("Testing inference for iid von mises model.")
    for kappa_true in [1, 10, 100]:
        for loc_true in np.array([0, -0.5, 0.5]) * np.pi:
            angles = vonmises.rvs(kappa_true, loc=loc_true, size=T)
            params_learned = estimate_von_mises_params(angles, VonMisesModelType.IID)
            print(f"True loc: {loc_true:.02f}, Estimated: {params_learned.drift:.02f}")
            print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
            assert np.isclose(params_learned.kappa, kappa_true, rtol=0.20)
            assert np.isclose(params_learned.drift, loc_true, atol=np.pi / 16)


def test_that__estimate_von_mises_params__gives_approximately_correct_concentration_parameter_for_a_von_mises_random_walk():
    T = 1000
    print("Testing inference for von mises random walk WITHOUT drift.")
    for kappa_true in [1, 10, 100]:
        angles = sample_from_von_mises_random_walk(kappa_true, T, init_angle=0.0)
        params_learned = estimate_von_mises_params(angles, VonMisesModelType.RANDOM_WALK)
        print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
        assert np.isclose(params_learned.kappa, kappa_true, rtol=0.20)


def test_that__estimate_von_mises_params__gives_approximately_correct_parameters_for_a_von_mises_random_walk_with_drift():
    T = 1000
    true_drift_angle = np.pi / 4
    init_angle = 0.0
    print("Testing inference for von mises random walk WITH drift.")
    for kappa_true in [1, 10, 100]:
        angles = sample_from_von_mises_random_walk_with_drift(
            kappa_true, T, init_angle, true_drift_angle
        )
        params_learned = estimate_von_mises_params(angles, VonMisesModelType.RANDOM_WALK_WITH_DRIFT)
        print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
        print(f"True drift: {true_drift_angle:.02f}, Estimated: {params_learned.drift:.02f}")
        assert np.isclose(params_learned.kappa, kappa_true, rtol=0.20)
        assert np.isclose(params_learned.drift, true_drift_angle, rtol=0.20)


def test_that__estimate_von_mises_params__gives_approximately_correct_parameters_for_a_von_mises_autoregression():
    T = 500
    drift_angles_true = np.array([-0.75, 0.0, 0.75]) * np.pi

    kappa_true = 10.0
    ar_coefs_true = np.array([-0.99, 0.0, 0.99])

    for ar_coef_true in ar_coefs_true:
        for drift_angle_true in drift_angles_true:
            print("\n---Next test on inference for von mises autoregression.")
            init_angle = drift_angle_true
            angles = sample_from_von_mises_AR_with_drift(
                kappa_true, T, ar_coef_true, init_angle, drift_angle_true
            )
            params_learned = estimate_von_mises_params(angles, VonMisesModelType.AUTOREGRESSION)
            print(f"ar coef true:{ar_coef_true:.02f}, learned:{params_learned.ar_coef:.02f}")
            print(f"True kappa: {kappa_true:.02f}, Estimated: {params_learned.kappa:.02f}")
            print(f"True drift: {drift_angle_true:.02f}, Estimated: {params_learned.drift:.02f}")
            assert np.isclose(params_learned.ar_coef, ar_coef_true, atol=0.30)
            assert two_angles_are_close(params_learned.drift, drift_angle_true, atol=np.pi / 8)
            assert np.isclose(params_learned.kappa, kappa_true, rtol=0.30)


def test_that__estimate_von_mises_params__handles_sample_weights__for_von_mises_autoregression():
    """
    We create a dataset whose first half is very different from the second half.
    We then see if we get very similar results when:
        - we omit the second half
        - we analyze the whole dataset, but put all the sample weight on the first half.
    """
    T_half = 100

    first_angles = sample_from_von_mises_AR_with_drift(
        kappa=10, T=T_half, ar_coef=0.2, init_angle=0.0, drift_angle=np.pi / 8
    )
    second_angles = sample_from_von_mises_AR_with_drift(
        kappa=100, T=T_half, ar_coef=-0.5, init_angle=np.pi, drift_angle=-np.pi / 8
    )
    angles = np.hstack((first_angles, second_angles))

    print("Testing use of sample weights when doing inference with von mises autoregression.")

    params_learned_first_half_via_sample_weights = estimate_von_mises_params(
        angles,
        VonMisesModelType.AUTOREGRESSION,
        sample_weights=np.hstack((np.ones(T_half), np.zeros(T_half))),
    )
    params_learned_first_half_via_omission = estimate_von_mises_params(
        angles[:T_half], VonMisesModelType.AUTOREGRESSION
    )
    assert two_angles_are_close(
        params_learned_first_half_via_omission.drift,
        params_learned_first_half_via_sample_weights.drift,
        atol=0.01,
    )
    assert np.isclose(
        params_learned_first_half_via_omission.ar_coef,
        params_learned_first_half_via_sample_weights.ar_coef,
        atol=0.01,
    )
    assert np.isclose(
        params_learned_first_half_via_omission.kappa,
        params_learned_first_half_via_sample_weights.kappa,
        atol=0.01,
    )
