import jax.numpy as jnp
import matplotlib.pyplot as plt
import seaborn as sns
from dynamax.utils.plotting import gradient_cmap


"""
vector fields inspired by dynamax 
https://github.com/probml/dynamax/blob/main/docs/notebooks/hmm/autoregressive_hmm.ipynb
"""

###
# Configs
###
params = params_learned
J, K = 5, 5

###
# Helper functions
###
sns.set_style("white")
sns.set_context("talk")

color_names = [
    "windows blue",
    "red",
    "amber",
    "faded green",
    "dusty purple",
    "orange",
    "brown",
    "pink",
]

colors = sns.xkcd_palette(color_names)
cmap = gradient_cmap(colors)


###
# Main Function
###


EPSILON = 0.2
X_MIN, X_MAX = 0.0, 2.0
Y_MIN, Y_MAX = 0.0, 1.0
x = jnp.linspace(X_MIN - EPSILON, X_MAX + EPSILON, 10)
y = jnp.linspace(Y_MIN - EPSILON, Y_MAX + EPSILON, 10)
X, Y = jnp.meshgrid(x, y)
xy = jnp.column_stack((X.ravel(), Y.ravel()))

fig, axs = plt.subplots(J, K, figsize=(12, 8))
for j in range(J):
    for k in range(K):
        A, b = params.CSP.As[j, k], params.CSP.bs[j, k]

        dxydt_m = xy.dot(A.T) + b - xy
        axs[j, k].quiver(
            xy[:, 0], xy[:, 1], dxydt_m[:, 0], dxydt_m[:, 1], color=colors[k % len(colors)]
        )

        axs[j, k].set_xlabel("$x_1$")
        axs[j, k].set_xticks([])
        if k == 0:
            axs[j, k].set_ylabel("$x_2$")
        axs[j, k].set_yticks([])
        axs[j, k].set_aspect("equal")


plt.tight_layout()
plt.show()

"""
TODOS:
- How do team states modulate between these?
"""
