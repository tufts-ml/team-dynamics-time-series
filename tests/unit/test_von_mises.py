import numpy as np

from dynagroup.von_mises.core import (
    VonMisesModelType,
    angles_from_points,
    estimate_von_mises_params,
    points_from_angles,
    sample_from_von_mises_random_walk,
)


def test_that__estimate_von_mises_params__gives_approximately_correct_concentration_parameter_for_a_von_mises_random_walk():
    T = 1000
    for kappa_true in [1, 10, 100]:
        print(
            f"Now estimating the concentration parameter, kappa, from a von Mises random walk where true kappa={kappa_true:.02f}"
        )
        angles = sample_from_von_mises_random_walk(kappa_true, T, init_angle=0.0)
        params_learned = estimate_von_mises_params(angles, VonMisesModelType.RANDOM_WALK)
        print(f"True kappa: {kappa_true}, Estimated: {params_learned.kappa:.02f}")
        assert np.isclose(params_learned.kappa, kappa_true, rtol=0.20)


def test__points_from_angles__then__angles_from_points():
    angles = np.array([-1, -0.5, 0, 0.5, 1]) * np.pi
    points = points_from_angles(angles)
    angles_reconstructed = angles_from_points(points)
    assert np.allclose(angles, angles_reconstructed)
