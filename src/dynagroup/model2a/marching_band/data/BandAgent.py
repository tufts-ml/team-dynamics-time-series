import numpy as np
import random

class BandAgent(object):

	def __init__(self, x_pos, y_pos, x_dir, ygrid_G, xgrid_H,cur_clum_state):
		self.x = float(x_pos)
		self.y = float(y_pos)
		self.y_ideal = float(y_pos)
		self.x_dir = int(x_dir)
		self.ygrid_G = ygrid_G
		self.xgrid_H = xgrid_H
		self.clum_state = int(cur_clum_state) # Whether the individual is currently out of bounds or not/in a clumsy state 


	def step(self, cur_state_GH, delta_x, sigma_x, sigma_y, prng):
		''' One step forward in time
		'''
		# Find current location in the global state map
		# As integer positions
		hh = np.searchsorted(self.xgrid_H, self.x) #current index for x 
		gg = np.searchsorted(self.ygrid_G, self.y_ideal) #current index for y
		G, H = cur_state_GH.shape # 100 x 100 grid size 

		B = 3 # num spaces to look ahead for "local" view for movement 
		hh_lo = np.maximum(hh-B, 0) # 3 spaces left of current position 
		hh_hi = np.minimum(hh+B, H-1) # 3 spaces right of current position 
		

		try:
			# If we are in 1, but we see only 0 in front, turn around. 
			if self.x_dir == 1:
				if hh_lo <= hh:
					v_cur = np.max(cur_state_GH[gg, hh_lo:hh+1])  # Maximum value in the current "local" view
				else:
					v_cur = cur_state_GH[gg, hh]  # Current position value	

				if hh + 1 < cur_state_GH.shape[1]:
					v_ahead = np.min(cur_state_GH[gg, hh+1:][:B]) # Minimum value ahead within the "local" view
					try:
						v_far = np.max(cur_state_GH[gg, hh+B+1:]) # Maximum value far ahead
					except IndexError:
						v_far = -1
					except ValueError:
						v_far = -1
				else: 
					v_ahead = -1 # out of bounds
					v_far = -1


			elif self.x_dir == -1: # Moving to the left
				if hh <= hh_hi:
					v_cur = np.max(cur_state_GH[gg, hh:hh_hi+1])    # Maximum value in the current "local" view
				else:
					v_cur = cur_state_GH[gg, hh] # Current position value
				if hh > 0:
					v_ahead = np.min(cur_state_GH[gg, :hh][::-1][:B]) # Minimum value in the current "local" view
					try:
						v_far = np.max(cur_state_GH[gg, :(hh-B-1)][::-1])  # Maximum value far ahead
					except IndexError:
						v_far = -1
					except ValueError:
						v_far = -1
				else:
					v_ahead = -1 # out of bounds
					v_far = -1

			if v_ahead == -1 and self.clum_state == 0:                  # out of bounds
				self.x_dir *= -1
			elif v_ahead == -1 and self.clum_state == 1:                  # out of bounds
				self.x_dir *= 1
			elif v_ahead == 0 and v_cur == 1:  # at end of local segment
				self.x_dir *= -1
			elif v_ahead == 0 and v_far <= 0 and v_cur == 0: # nothing this dir
				self.x_dir *= -1 # turn around

		except IndexError:
			# We're already at edge, turn around
			self.x_dir *= -1


		self.x += self.x_dir * delta_x

		# Add noise
		self.x += sigma_x * prng.random()
		self.y += sigma_y * prng.random()

