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


COLORS = [
    '#a6cee3',
    '#1f78b4',
    '#b2df8a',
    '#33a02c',
    '#fb9a99',
    '#e31a1c',
    '#fdbf6f',
    '#ff7f00',
    '#cab2d6',
    '#6a3d9a',
    '#ffff99',
    '#b15928',
]

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
        # flip upside down so y indexing works with 0 as origin
        STATEMAPS[k] = np.asarray(np.flipud(v_GH) > 0.01, dtype=np.int32) 

    prng = np.random.default_rng(seed)

    delta_N = prng.uniform(low=0.015, high=0.05, size=N)
    color_N = prng.choice(COLORS, size=N, replace=True)

    agents = []
    for n in range(N):
        xstart = prng.uniform(low=0, high=0.05)
        ystart = np.linspace(0, 1, N)[n]
        a = BandAgent(xstart, ystart, 1, ygrid_G, xgrid_H, cur_clum_state)
        agents.append(a)


    U = len(GLOBAL_MSG) * T + (50 * 30)
    pos_NU2 = np.zeros((U, N, 2))
    threshold = 7 #The amount of players that can become clumsy before the coach calls a cluster state. 

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
                    agents[n].clum_state = prng.choice([0, 1], p=[0.9999, 0.0001]) 
    
            
            k_list = [agents[n].clum_state for n in range(N)]
            k = np.sum(k_list)
            
            if k > threshold: 
                for n in range(N): 
                    if agents[n].x > 1: 
                        agents[n].x = agents[n].x - 1 
                    if agents[n].x < 0: 
                        agents[n].x = agents[n].x + 1
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


            if t > 3 and t % 5 == 0:
                if k <= threshold: 
                    plt.title(cur_state, fontsize=14)
                else: 
                    plt.title('C', fontsize=14)
                for n in range(N):
                    marker = '>' if agents[n].x_dir == 1 else '<'
                    plt.plot(pos_NU2[uu-5:uu,n,0], pos_NU2[uu-5:uu,n,1],
                        '.-', color=color_N[n])
                    plt.plot(pos_NU2[uu,n,0], pos_NU2[uu,n,1],
                        marker, color=color_N[n])
                ax = plt.gca()
                ax.set_xlim([0,1])
                ax.set_ylim([0,1])
                ax.set_aspect('equal')
                plt.show(block=False)
                path = f'/Users/kgili/team-dynamics-time-series/src/dynagroup/model2a/marching_band/data/frames/{seed}'
                os.makedirs(path, exist_ok=True)
                fpath = os.path.join(path, 'step%05d.png' % step)
                plt.savefig(fpath)
                plt.clf()
                step += 1

                if step < 3 or step % 20 == 0 or step == U//5:
                    print('%s step %5d after %.1f sec' % (cur_state, step, time.time()-starttime))

        
        if ss > 0 and ((ss+6) % 5 == 0): 
            sequence_end_times.append(uu + 1)
    pos_NU2 = remove_zeros(pos_NU2)
    return pos_NU2, sequence_end_times, trigger_index_list


def remove_zeros(data):

    non_zero_subarrays = ~np.all(data == 0, axis=(1, 2))
    filtered_arr = data[non_zero_subarrays]
    return filtered_arr


def plot_segmentation_gt(data, sequence): 

    sequence_end_times = data[1]
    trigger_index_list = data[2]

    start = sequence_end_times[sequence-1]
    end = sequence_end_times[sequence]

         
    #ALWAYS START WITH NORMAL SEGMENTS 200, 400,.. AND THEN ADD IN WHERE THE CLUSTER STATES ARE
    segments = [
        {"label": "L", "start": start, "end": start + 200, "color": '#ff7f00'},
        {"label": "A", "start": start + 200, "end": start + 450, "color": '#cab2d6'},
        {"label": "U", "start": start + 450, "end": start + 650, "color": '#6a3d9a'},
        {"label": "G", "start": start + 650, "end": start + 850, "color": '#ffff99'},
        {"label": "H", "start": start + 850, "end": start + 1050, "color": '#b15928'}
    ]    

    for j in trigger_index_list: 
        if j <= end and j >= start: 
            segments.append({"label": "C", "start": j, "end": j+50, "color": '#a6cee3'})
    
    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))

    # Plot each segment
    for segment in segments:
        rect = patches.Rectangle((segment["start"], 0), segment["end"] - segment["start"], 1, 
                                linewidth=1, edgecolor='black', facecolor=segment["color"])
        ax.add_patch(rect)

    # Add labels and grid
    ax.set_yticks([])
    # ax.set_xticks(np.arange(0, 121, 10))
    # ax.set_xticklabels(np.arange(0, 121, 10))
    ax.set_xlim(start, end)
    # ax.set_ylim(0, 1)

    # Add a legend
    handles = [patches.Patch(color=segment["color"], label=segment["label"]) for segment in segments]
    ax.legend(handles=handles, loc='upper right')

    # Title and show plot
    plt.title('System State Segmentation')
    plt.show()

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
    data = array1[0]
    trigger = array1[2]
    x = system_regimes_gt(10, [1016, 2479, 4182, 6195, 7341, 8965])
    print(x)
    
    
