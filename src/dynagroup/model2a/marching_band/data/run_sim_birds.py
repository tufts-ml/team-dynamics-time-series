'''
Script to simulate many agents forming a sequence of letters

Usage
-----
Run at command line with no arguments

Expected output
---------------
frames/ folder filled with .png images from throughout the sim
'''

import numpy as np
import skimage.transform
import matplotlib.pyplot as plt;
import time, os
from random import choices
import matplotlib.patches as patches

from dynagroup.model2a.marching_band.data.BandAgentwithBirds import get_noisy_bird_positions_from_band_agents, BandAgent
from dynagroup.model2a.marching_band.data.templates import STATEMAP_ARRAYS_BY_NAME


home_dir = os.path.expanduser("~")


def remove_zeros(data):
    """
    Keep only time steps that have non-zero agent observations.
    Checks all agents at a given time step. If that entire "slice" is zero, remove it.
    """
    # data has shape (U, N, obs_dim). We check if an entire row is all zeros.
    non_zero_subarrays = ~np.all(data == 0, axis=(1, 2))
    filtered_arr = data[non_zero_subarrays]
    return filtered_arr

def generate_bird_training_data(
    GLOBAL_MSG, 
    N, 
    T, 
    seed=0, 
    num_birds_per_agent=3,  # <<-- New parameter
    noise_std=0.02
):
    """
    Generate training data where each of N agents is simulated across
    len(GLOBAL_MSG)*T time steps. At each step, each agent yields multiple
    noisy 'bird' observations. The final returned data is only those noisy birds.

    Args:
    - GLOBAL_MSG: string message sequence (e.g., 'LAUGHLAUGH...')
    - N: number of band agents
    - T: number of time steps per letter in GLOBAL_MSG
    - seed: random seed for reproducibility
    - num_birds_per_agent: how many birds to generate for each agent at each time step
    - noise_std: standard deviation of the Gaussian noise for each bird's (x,y)

    Returns:
    - obs_NU: shape (U, N, 2 * num_birds_per_agent)
        Where U = len(GLOBAL_MSG)*T.
        For each time step U and each agent N, we store a flattened vector
        of length 2*M: [bird1_x, bird1_y, bird2_x, bird2_y, ..., birdM_x, birdM_y]
    - sequence_end_times: optional list of end indices if you want them
    """

    G = 100
    H = 100
    prng = np.random.default_rng(seed)
    ygrid_G = np.linspace(0, 1, G)
    xgrid_H = np.linspace(0, 1, H)
    cur_clum_state = 0

    agents = []
    for n in range(N):
        xstart = prng.uniform(low=0, high=0.05)
        ystart = np.linspace(0, 1, N)[n]
        a = BandAgent(xstart, ystart, 1, ygrid_G, xgrid_H, cur_clum_state)
        agents.append(a)

    STATEMAPS = {}
    for k, v_AB in STATEMAP_ARRAYS_BY_NAME.items():
        v_GH = skimage.transform.resize(
            v_AB, (G, H), mode='constant', preserve_range=True
        )
        # Flip and threshold
        STATEMAPS[k] = np.asarray(np.flipud(v_GH) > 0.01, dtype=np.int32)


    delta_N = prng.uniform(low=0.015, high=0.05, size=N)

  
    U = len(GLOBAL_MSG) * T

    # (Optional) track the agent’s true positions for debugging 
    pos_NU2 = np.zeros((U, N, 2))


    obs_dim = 2 * num_birds_per_agent
    obs_NU = np.zeros((U, N, obs_dim))

    # A small helper function to get M noisy birds for a single agent
    def get_multiple_birds_for_agent(agent, rng, M, noise_std):
        """
        Returns a flattened 1D array of length 2*M:
          [bx1, by1, bx2, by2, ..., bxM, byM]
        """
        # Each "bird" is N( agent.x, noise_std ), N( agent.y, noise_std )
        # Then clipped to [0,1].
        birds_xy = np.zeros((M, 2))
        for i in range(M):
            bx = rng.normal(loc=agent.x, scale=noise_std)
            by = rng.normal(loc=agent.y, scale=noise_std)
            bx = np.clip(bx, 0.0, 1.0)
            by = np.clip(by, 0.0, 1.0)
            birds_xy[i, 0] = bx
            birds_xy[i, 1] = by
        # Flatten (M,2) -> shape (2*M,)
        return birds_xy.ravel()

    print(f"Running simulation for {U} steps...")
    step = 0
    starttime = time.time()
    sequence_end_times = [-1]


    for ss, cur_state_key in enumerate(GLOBAL_MSG):

        # 6A) Grab the current state map (100x100)
        cur_statemap = STATEMAPS[cur_state_key]

        for t in range(T):
            uu = ss*T + t  # overall time index

            # 6B) Step each agent
            for n in range(N):
                agents[n].step(
                    cur_statemap,   # 100x100 map
                    delta_N[n],     # agent-specific delta_x
                    0.004,          # sigma_x
                    0.0015,         # sigma_y
                    prng
                )
                # Optional: store agent’s *true* position 
                pos_NU2[uu, n, 0] = agents[n].x
                pos_NU2[uu, n, 1] = agents[n].y

            # 6C) Now, for each agent, generate M noisy birds
            for n in range(N):
                obs_NU[uu, n, :] = get_multiple_birds_for_agent(
                    agent=agents[n],
                    rng=prng,
                    M=num_birds_per_agent,
                    noise_std=noise_std
                )

        # Optionally mark sequence end times
        if ss > 0 and ((ss+6) % 5 == 0):
            sequence_end_times.append(uu + 1)

  
    obs_NU = remove_zeros(obs_NU)

    # Return just the birds in obs_NU (and end times if you want them)
    return obs_NU, sequence_end_times

# ----------------------------------------------------------------
# Example usage:
if __name__ == '__main__':
    GLOBAL_MSG = 'LAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGH'
    N = 30
    T = 200

    # Let's say we now want 5 birds per agent instead of just 1
    num_birds_per_agent = 10
    obs_data, end_times = generate_bird_training_data(
        GLOBAL_MSG=GLOBAL_MSG,
        N=N,
        T=T,
        seed=0,
        num_birds_per_agent=num_birds_per_agent,
        noise_std=0.02  # standard deviation for noise
    )

    