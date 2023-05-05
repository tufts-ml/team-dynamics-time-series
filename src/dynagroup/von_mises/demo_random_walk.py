import numpy as np
from matplotlib import pyplot as plt

from dynagroup.von_mises.generate import sample_from_von_mises_random_walk
from dynagroup.von_mises.inference import VonMisesModelType, estimate_von_mises_params


"""
We get iid samples from a von Mises random walk and then do inference.
"""

###
# Sample a Von Mises distribution
###

T = 1000
kappa_true = 20
angles = sample_from_von_mises_random_walk(kappa_true, T, init_angle=0.0)

###
# Plot samples
###

plt.scatter(np.arange(T), angles, c=np.arange(T), cmap="cool")
plt.xlabel("time")
plt.ylabel(r"angle $\in [-\pi, \pi]$")
plt.tight_layout()
plt.show()

# points=points_from_angles(angles)
# plt.scatter(points[:,0], points[:,1], c=np.arange(len(points)), cmap="cool")
# plt.show()


####
# Estimate parameters
###

params_learned = estimate_von_mises_params(angles, VonMisesModelType.RANDOM_WALK)
print(f"True kappa: {kappa_true}, Estimated: {params_learned.kappa:.02f}")
