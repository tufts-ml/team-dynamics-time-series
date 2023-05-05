import numpy as np
from scipy.stats import vonmises

from dynagroup.von_mises.core import (
    VonMisesModelType,
    angles_from_points,
    estimate_von_mises_params,
    points_from_angles,
    sample_from_von_mises_random_walk,
    sample_from_von_mises_random_walk_with_drift,
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
