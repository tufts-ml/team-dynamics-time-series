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

from dynagroup.model2a.marching_band.data.BandAgent import BandAgent
from dynagroup.model2a.marching_band.data.templates import STATEMAP_ARRAYS_BY_NAME


home_dir = os.path.expanduser("~")

def generate_training_data(GLOBAL_MSG, N, T, seed): 
    G = 100
    H = 100
    xgrid_H = np.linspace(0, 1, H)
    ygrid_G = np.linspace(0, 1, G)
    cur_clum_state = 0

    STATEMAPS = {}
    for k, v_AB in STATEMAP_ARRAYS_BY_NAME.items():
        v_GH = skimage.transform.resize(
            v_AB, (G, H), mode='constant', preserve_range=True)
      
        STATEMAPS[k] = np.asarray(np.flipud(v_GH) > 0.01, dtype=np.int32) 

    prng = np.random.default_rng(seed)

    delta_N = prng.uniform(low=0.015, high=0.05, size=N)
    COLORS = prng.choice(['#ff7f00','#cab2d6','#6a3d9a','#ffff99','#b15928','#a6cee3'], size=N) 

    agents = []
    for n in range(N):
        xstart = prng.uniform(low=0, high=0.05)
        ystart = np.linspace(0, 1, N)[n]
        a = BandAgent(xstart, ystart, 1, ygrid_G, xgrid_H, cur_clum_state)
        agents.append(a)


    U = (len(GLOBAL_MSG) * T) + (50 * 50)
    pos_NU2 = np.zeros((U, N, 2))
    threshold = 11 #The amount of players that can go out of bounds before a cluster state is triggered. 

    print("Running simulation for %d steps, recording every 5th step" % U)

    step = 0
    starttime = time.time()
    trigger_index_list = []
    sequence_end_times = [-1]
    c = 0

    for ss, cur_state in enumerate(GLOBAL_MSG):
          

        for t in range(T):
        
            uu = ss*T + t + (50*c)

            for n in range(N):
                if agents[n].clum_state == 1: 
                    pass
                else: 
                    agents[n].clum_state = prng.choice([0, 1], p=[0.99999, 0.00001]) 
    
            k = 0
            k_list = [agents[n].x for n in range(N)]
            for elem in k_list: 
                if elem > 1 or elem < 0: 
                    k+= 1 
            
            if k > threshold: 
                for n in range(N): 
                    if agents[n].x > 1: 
                        agents[n].x = 0.5
                    if agents[n].x < 0: 
                        agents[n].x = 0.5
                    agents[n].clum_state = 0
               
                trigger_index_list.append(uu)    

                for l in range(51): 
                    for n in range(N):
                        agents[n].step(STATEMAPS['cluster'], delta_N[n], 0.004, 0.0015, prng)
                        pos_NU2[l + uu , n, 0] = agents[n].x
                        pos_NU2[l + uu , n, 1] = agents[n].y
                c+= 1

            else: 
                for n in range(N):
                    agents[n].step(STATEMAPS[cur_state], delta_N[n], 0.004, 0.0015, prng)
                    pos_NU2[uu, n, 0] = agents[n].x
                    pos_NU2[uu, n, 1] = agents[n].y


            # if t > 3 and t % 5 == 0:
            #     if k <= threshold: 
            #         plt.title(cur_state, fontsize=14)
            #     else: 
            #         plt.title('C', fontsize=14)
            #     for n in range(N):
            #         marker = '>' if agents[n].x_dir == 1 else '<'
            #         plt.plot(pos_NU2[uu-5:uu,n,0], pos_NU2[uu-5:uu,n,1],
            #             '.-', color=COLORS[n])
            #         plt.plot(pos_NU2[uu,n,0], pos_NU2[uu,n,1],
            #             marker, color=COLORS[n])
            #     ax = plt.gca()
            #     ax.set_xlim([0,1])
            #     ax.set_ylim([0,1])
            #     ax.set_aspect('equal')
            #     plt.show(block=False)
            #     path = f'{home_dir}/team-dynamics-time-series/src/dynagroup/model2a/marching_band/data/frames/{seed}'
            #     os.makedirs(path, exist_ok=True)
            #     fpath = os.path.join(path, 'step%05d.png' % step)
            #     plt.savefig(fpath)
            #     plt.clf()
            #     step += 1

            #     if step < 3 or step % 20 == 0 or step == U//5:
            #         print('%s step %5d after %.1f sec' % (cur_state, step, time.time()-starttime))
        
        if ss > 0 and ((ss+6) % 5 == 0): 
            sequence_end_times.append(uu + 1)
    pos_NU2 = remove_zeros(pos_NU2)
    return pos_NU2, sequence_end_times, trigger_index_list


def remove_zeros(data):
    non_zero_subarrays = ~np.all(data == 0, axis=(1, 2))
    filtered_arr = data[non_zero_subarrays]
    return filtered_arr


def system_regimes_gt(num_sequences, trigger): 
    og = num_sequences*1000
    system_regimes = np.zeros((og, 6)) 
    segments = [i for i in range(0, (num_sequences*1000)+ 200, 200)]
    for t, i in enumerate(segments[:-1]): 
        for j in range(og):
            if j < segments[t+1] and j >= segments[t]: 
                system_regimes[j][t%5] = 1
    
    trigger_regimes = np.zeros((50, 6)) 
    for i in range(50): 
        trigger_regimes[i][5] = 1

    for i in trigger: 
        part1 = system_regimes[:i]
        part3 = system_regimes[i:]
        system_regimes = np.vstack((part1, trigger_regimes, part3))

    return system_regimes


if __name__ == '__main__':
    GLOBAL_MSG = 'LAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGHLAUGH'
    N = 64
    T = 200
    array1 = generate_training_data(GLOBAL_MSG, N, T, 0)


    
   