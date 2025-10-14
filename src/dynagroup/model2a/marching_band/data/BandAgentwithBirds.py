import numpy as np

class BandAgent(object):
    def __init__(self, x_pos, y_pos, x_dir, ygrid_G, xgrid_H, cur_clum_state):
        self.x = float(x_pos)
        self.y = float(y_pos)
        self.y_ideal = float(y_pos)
        self.x_dir = int(x_dir)
        self.ygrid_G = ygrid_G
        self.xgrid_H = xgrid_H
        self.clum_state = int(cur_clum_state)

    def step(self, cur_state_GH, delta_x, sigma_x, sigma_y, prng):
        """
        One step forward in time. Then clip (x,y) to [0,1].
        """
        hh = np.searchsorted(self.xgrid_H, self.x)
        gg = np.searchsorted(self.ygrid_G, self.y_ideal)
        G, H = cur_state_GH.shape

        B = 3
        hh_lo = np.maximum(hh - B, 0)
        hh_hi = np.minimum(hh + B, H - 1)

        try:
            if self.x_dir == 1:
                # moving right
                if hh_lo <= hh:
                    v_cur = np.max(cur_state_GH[gg, hh_lo:hh+1])
                else:
                    v_cur = cur_state_GH[gg, hh]

                if hh + 1 < cur_state_GH.shape[1]:
                    v_ahead = np.min(cur_state_GH[gg, hh+1:][:B])
                    try:
                        v_far = np.max(cur_state_GH[gg, hh+B+1:])
                    except (IndexError, ValueError):
                        v_far = -1
                else:
                    v_ahead = -1
                    v_far = -1

            else:
                # moving left
                if hh <= hh_hi:
                    v_cur = np.max(cur_state_GH[gg, hh:hh_hi+1])
                else:
                    v_cur = cur_state_GH[gg, hh]

                if hh > 0:
                    v_ahead = np.min(cur_state_GH[gg, :hh][::-1][:B])
                    try:
                        v_far = np.max(cur_state_GH[gg, :(hh-B-1)][::-1])
                    except (IndexError, ValueError):
                        v_far = -1
                else:
                    v_ahead = -1
                    v_far = -1

            # Logic for direction changes
            if v_ahead == -1 and self.clum_state == 0:
                self.x_dir *= -1
            elif v_ahead == -1 and self.clum_state == 1:
                self.x_dir *= 1
            elif v_ahead == 0 and v_cur == 1:
                self.x_dir *= -1
            elif v_ahead == 0 and v_far <= 0 and v_cur == 0:
                self.x_dir *= -1

        except IndexError:
            # at the edge
            self.x_dir *= -1

        # Update position
        self.x += self.x_dir * delta_x
        # Add small noise
        self.x += sigma_x * prng.random()
        self.y += sigma_y * prng.random() * 0.001

        # Clip to remain in [0,1]
        self.x = np.clip(self.x, 0.0, 1.0)
        self.y = np.clip(self.y, 0.0, 1.0)

def get_noisy_bird_positions_from_band_agents(
    band_agents,
    noise_std=0.02,
    seed=0
):
    """
    Generate exactly one 'bird' for each band agent by sampling around
    each agent's position with Gaussian noise.

    - band_agents: list of BandAgent objects
    - noise_std: std dev of Gaussian noise around each agent's position
    - seed: random seed for reproducibility

    Returns:
        A NumPy array of shape (len(band_agents), 2) containing the
        (x, y) of the "bird" observations.
    """
    prng = np.random.RandomState(seed)

    birds = []
    for agent in band_agents:
        # Add Gaussian noise to the agent's true position
        x_noisy = prng.normal(loc=agent.x, scale=noise_std)
        y_noisy = prng.normal(loc=agent.y, scale=noise_std)

        # Clip so we remain in [0,1]
        x_noisy = np.clip(x_noisy, 0.0, 1.0)
        y_noisy = np.clip(y_noisy, 0.0, 1.0)

        birds.append((x_noisy, y_noisy))

    return np.array(birds)

