import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import bnpy

def load_data_for_playerid(playerid,
        datafilepath='player_coords_train__with_1_games.npy',
        boxpath='/Users/mhughes/Box//Group_Dynamics_Data/version_1.1.0.20230823/'):
    '''

    Returns
    -------
    X
    doc_range : pointer of start/stop ids
    '''
    stopsfilepath = datafilepath.replace('player_coords', 'play_start_stop_idxs')
    Q = np.load(os.path.join(boxpath, datafilepath))
    stops = np.load(os.path.join(boxpath, stopsfilepath))
    rawx_N2 = Q[:, playerid, :]
    doc_range_M = np.hstack([0, stops[:,1].copy()])
    doc_range_M[0] = 0
    assert np.all(np.diff(doc_range_M) > 0)

    xcur_list = list()
    xprev_list = list()
    for start, stop in zip(doc_range_M[:-1], doc_range_M[1:]):
         if stop - start < 10:
            continue
         xseq_N2 = rawx_N2[start:stop]
         xcur_N2 = xseq_N2[1:]
         xprev_N2 = xseq_N2[:-1]
         xcur_list.append(xcur_N2)
         xprev_list.append(xprev_N2)
    x_N2 = np.vstack(xcur_list)
    xprev_N3 = np.hstack([
        np.vstack(xprev_list),
        np.ones((x_N2.shape[0], 1))])

    len_S = np.asarray([len(xx) for xx in xcur_list])
    doc_range_S = np.hstack([0, np.cumsum(len_S)])

    return bnpy.data.GroupXData(
        x_N2, doc_range=doc_range_S, Xprev=xprev_N3)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--hmmKappa', type=float, default=100.)
    parser.add_argument('--sF', type=float, default=0.1)
    parser.add_argument('--nTask', type=int, default=20)
    parser.add_argument('--output_path', type=str,
        default='/tmp/')
    args = parser.parse_args()

    for key, val in args.__dict__.items():
        print("--%s %s" % (key, val))

    K = args.K
    hmmKappa = args.hmmKappa
    sF = args.sF
    nTask = args.nTask
    nLap1 = 20
    nLap2 = 900

    data = {}
    for key, suffix in [
        ('train', 'train__with_5_games.npy'),
        ('valid', 'val__with_4_games.npy'),
        ('test', 'test__with_5_games.npy'),
        ]:
        data[key] = load_data_for_playerid(0,
            datafilepath='player_coords_%s' % suffix)

    output_path = os.path.join(output_path,
        'arhmm-K=%03d-hmmKappa=%03d-sF=%5.2f' % (
            K, hmmKappa, sF))
    output_path_extended = output_path + '-extended_run'

    model1, info1 = bnpy.run(
        data['train'], 'FiniteHMM', 'AutoRegGauss', 'EM',
        K=K,
        nTask=nTask, nLap=nLap1, initname='randcontigblocks',
        initBlockLen=20, # 4 sec chunks
        output_path=output_path,
        convergeThr=0.0001,
        startAlpha=10.0, transAlpha=0.5, hmmKappa=hmmKappa,
        MMat='zero', ECovMat='eye', sF=sF,
        printEvery=20, saveEvery=10)
    model2, info2 = bnpy.run(
        data['train'], 'FiniteHMM', 'AutoRegGauss', 'EM',
        K=K,
        nTask=1, nLap=nLap2, initname=info1['task_output_path'],
        output_path=output_path_extended
        convergeThr=0.000001,
        startAlpha=10.0, transAlpha=0.5, hmmKappa=hmmKappa,
        MMat='zero', ECovMat='eye', sF=0.1,
        printEvery=20, saveEvery=10)

