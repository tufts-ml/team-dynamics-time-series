'''
Compute entropy of a variational posterior of an HMM

Doctests
--------
>>> T = 2; U = 1; L = 2;
>>> s_ULL = np.asarray([[[0.4, 0.3], [0.2, 0.1]]])
>>> r_TL = np.asarray([[0.7, 0.3], [0.6, 0.4]])
>>> h = calc_entropy_hmm_posterior(r_TL, s_ULL, do_assert_input_valid=True)
>>> np.round(h.item(), 5)
1.27985
>>> -1 * np.round(np.sum(s_ULL * np.log(s_ULL)), 5) # verify another way
1.27985

>>> s_ULL = np.asarray([[[0.996, 0.002], [0.001, 0.001]]])
>>> r_TL = np.asarray([[0.998, 0.002], [0.997, 0.003]])
>>> h = calc_entropy_hmm_posterior(r_TL, s_ULL, do_assert_input_valid=True)
>>> np.round(h.item(), 5)
0.03024
>>> -1 * np.round(np.sum(s_ULL * np.log(s_ULL)), 5) # verify another way
0.03024

>>> T = 5; U = 4; L = 2
>>> s_ULL = np.asarray([
...     [[0.999, 0. ],
...      [0.001, 0. ]],
...     [[1. , 0. ],
...      [0. , 0. ]],
...     [[1. , 0. ],
...      [0. , 0. ]],
...     [[0. , 1. ],
...      [0. , 0. ]]])
>>> r_UL = np.sum(s_ULL, axis=2)
>>> r_TL = np.vstack([r_UL, s_ULL[-1].sum(axis=0, keepdims=1)])
>>> h = calc_entropy_hmm_posterior(r_TL, s_ULL, do_assert_input_valid=True)
>>> np.round(h.item(), 5)
0.00791
'''
import numpy as np
import jax.numpy as jnp

def calc_entropy_hmm_posterior(
        r_TL, s_ULL, do_assert_input_valid=False, eps=1e-13):
    ''' Calculate entropy of HMM hidden state sequence distribution

    Args
    ----
    r_TL : 2D array, shape (T, L)
        r_TL[t,l] := p( z[t] = l )
        Per-timestep marginal distribution over states
    s_ULL : 3D array, shape (T-1, L, L), where U = T-1
        s_ULL[t,k,l] := p( z[t] = k, z[t+1] = l)
        Joint distribution over states for adjacent tsteps t, t+1
        Strictly required that r is a marginal of s
        * r_TL[:-1] = sum(s_ULL, axis=2)
        * r_TL[-1]  = sum(s_ULL[-1], axis=0)

    Returns
    -------
    entropy : float
        Entropy of the provided distribution
    ''' 
    if do_assert_input_valid:
        assert jnp.allclose(r_TL[:-1], np.sum(s_ULL, axis=2)).item()
        assert jnp.allclose(r_TL[-1], np.sum(s_ULL[-1], axis=0)).item()
    # Entropy at time 0
    r0_L = r_TL[0]
    h0 = -jnp.sum(r0_L * jnp.log(r0_L + eps))

    # Entropy at times 1, 2, ... T-1
    # Defined as sum of conditional entropy
    # = H[ z_1 | z_0] + H[ z_2 | z_1] + ...

    # step 1, compute array c_ULL
    # where c_ULL[t, j, k] := p(z_t+1 = k | z_t = j)
    denom_UL1 = r_TL[:-1][:,:,np.newaxis]
    c_ULL = s_ULL / (eps + denom_UL1)
    # step 2, compute conditional entropy using its traditional formula
    h1toT_U = -jnp.sum(s_ULL * jnp.log(c_ULL + eps), axis=(1,2))
    return h0 + jnp.sum(h1toT_U)