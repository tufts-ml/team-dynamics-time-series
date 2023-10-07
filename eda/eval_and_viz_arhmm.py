import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import bnpy

from train_arhmm import load_data_for_playerid

def draw_forecast_given_context(
        model, xctx_T2, xprev_T3, Tfuture=16,
        prng=np.random):
    T = xctx_T2.shape[0]
    K = model.obsModel.EstParams.K
    assert xprev_T3.shape[1] == xctx_T2.shape[1] + 1
    data = bnpy.data.GroupXData(
        xctx_T2, 
        Xprev=xprev_T3,
        doc_range=np.asarray([0, T]),
        )
    LP = model.calc_local_params(data)

    # Sample state seq
    trans_proba_KK = model.allocModel.transPi
    A_K23 = model.obsModel.EstParams.A
    cov_K22 = model.obsModel.EstParams.Sigma

    b_0_K = LP['resp'][-1]
    z_T = np.zeros(Tfuture+1)
    x_T2 = np.zeros((Tfuture+1, 2))
    z_T[0] = -1 # mark as avoid
    x_T2[0] = xctx_T2[-1]
    proba_t_K = np.squeeze(np.dot(b_0_K[np.newaxis,:], trans_proba_KK))
    xprev_t_3 = np.hstack([xctx_T2[-1], 1.])
    for t in range(1, 1+Tfuture):
        z_t = prng.choice(np.arange(K), p=proba_t_K)
        z_T[t] = z_t
        proba_t_K = trans_proba_KK[z_t]

        x_t_2 = prng.multivariate_normal(
            np.dot(A_K23[z_t], xprev_t_3),
            cov_K22[z_t])

        D = 0.05
        if np.sqrt(np.sum(np.square(x_t_2 - x_T2[t-1]))) > D:
            vec = x_t_2 - x_T2[t-1]
            x_t_2 = x_T2[t-1] + vec * D / np.sqrt(np.sum(np.square(vec)))
        x_T2[t] = x_t_2
        xprev_t_3 = np.hstack([x_t_2, 1.])

        assert np.sqrt(np.sum(np.square(x_t_2 - x_T2[t-1]))) < 1.05 * D
    return x_T2, z_T

def eval_samples_on_dataset(
        data_key,
        data_obj,
        make_fresh_plot_grid,
        Tpast=32,
        Tfuture=16,
        n_samples=50):
    data_seed = int(np.sum(data_obj.doc_range))

    pp = 0
    dr_M = data_obj.doc_range
    n_test = dr_M.size - 1

    fig, axgrid = make_fresh_plot_grid()
    n_show = axgrid.flatten().size
    row_list = list()
    for seq_id in range(n_test):
        start, stop = dr_M[seq_id], dr_M[seq_id+1]
        if (stop - start) < (Tpast + Tfuture):
            continue
        xctx_T2 = data_obj.X[start:stop]
        xprev_T3 = data_obj.Xprev[start:stop]

        # Stationary baseline
        xstat_U2 = np.tile(xctx_T2[Tpast-1][np.newaxis,:], (Tfuture+1,1))

        # Last instantaneous velocity baseline
        vel_2 = xctx_T2[Tpast-1] - xctx_T2[Tpast-2]
        xvel_inst_U2 = xctx_T2[Tpast-1,:] + (
            vel_2[np.newaxis,:] * np.arange(0, Tfuture+1)[:,np.newaxis])
        
        # Last 2 second velocity baseline
        W = 10
        vel_2 = np.mean(np.diff(xctx_T2[Tpast-W:Tpast-1], axis=0), axis=0)
        xvel_2sec_U2 = xctx_T2[Tpast-1,:] + (
            vel_2[np.newaxis,:] * np.arange(0, Tfuture+1)[:,np.newaxis])

        ax = axgrid.flatten()[pp]
        xs = np.linspace(0, 1, 10)
        ys = np.linspace(0, 1, 10)
        ax.plot(xs[0] * np.ones_like(ys), ys, 'k-')
        ax.plot(xs[-1] * np.ones_like(ys), ys, 'k-')
        ax.plot(xs, ys[0] * np.ones_like(xs), 'k-')
        ax.plot(xs, ys[-1] * np.ones_like(xs), 'k-')
        
        prng = np.random.RandomState(100 * data_seed + seq_id)
        xsamp_SU2 = np.zeros((n_samples, 1+Tfuture, 2))
        zsamp_SU = np.zeros((n_samples, 1+Tfuture), dtype=np.int32)
        mae_S = np.zeros((n_samples,))
        for ss in range(n_samples):
            xsamp_SU2[ss], zsamp_SU[ss] = draw_forecast_given_context(
                model, xctx_T2[:Tpast], xprev_T3[:Tpast],
                Tfuture, prng)
            mae_S[ss] = np.mean(np.abs(
                xsamp_SU2[ss,1:] - xctx_T2[Tpast:Tpast+Tfuture]))

            if ss < n_show:
                ax.plot(xsamp_SU2[ss,:,0], xsamp_SU2[ss,:,1], 'k.-', alpha=0.4)

        metrics = {'seq_id':seq_id}
        for prctile in [5, 10, 25, 50]:
            metrics['mae_%02d' % (prctile)] = np.percentile(mae_S, prctile)
        for key, arr_U2 in [
                ('stat', xstat_U2),
                ('vel_inst', xvel_inst_U2),
                ('vel_2sec', xvel_2sec_U2)]:
            metrics['mae_%s' % key] = np.mean(np.abs(
                arr_U2[1:] - xctx_T2[Tpast:Tpast+Tfuture]))
        row_list.append(metrics)

        ax.plot(xvel_inst_U2[:Tpast,0], xvel_inst_U2[:Tpast,1], 'c.-', alpha=0.4)
        ax.plot(xctx_T2[:Tpast,0], xctx_T2[:Tpast,1], 'b.-')
        ax.plot(
            xctx_T2[Tpast:Tpast+Tfuture,0],
            xctx_T2[Tpast:Tpast+Tfuture,1], 'r.-')
        ax.set_xlim([-0.1, 1.1]);
        ax.set_ylim([-0.1, 1.1]);
        ax.set_xticks([]);
        ax.set_yticks([]);
        ax.set_title('MAE 10p:%5.3f\nMAE vel:%5.3f' % (
            metrics['mae_10'], metrics['mae_vel_inst']))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        pp += 1
        if pp >= n_show:
            plt.tight_layout()
            plt.show(block=False)
            plt.savefig(
                'results/%s_traj_samples_upto_seqid%02d.pdf' % (data_key, seq_id))
            pp = 0
            plt.close()
            fig, axgrid = make_fresh_plot_grid()
    metrics_df = pd.DataFrame(row_list)
    return metrics_df
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--K', type=int, default=10)
    parser.add_argument('--hmmKappa', type=float, default=100.)
    parser.add_argument('--sF', type=float, default=0.1)
    parser.add_argument('--nTask', type=int, default=20)
    args = parser.parse_args()

    for key, val in args.__dict__.items():
        print("--%s %s" % (key, val)

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

    model, l = bnpy.load_model_at_lap(
        '/tmp/nba_5game/arhmm-K=25-extended/1/', 600)

    # Prepare visuals
    W = 4
    H = 2.
    nrows, ncols = 3, 4
    def make_fresh_plot_grid():
        return plt.subplots(
            nrows=nrows, ncols=ncols, figsize=(W*ncols, H*nrows),
            )

    sum_list = list()
    for key in data.keys():
        metrics_for_cur_split_df = eval_samples_on_dataset(
            key, data[key], make_fresh_plot_grid)
        metrics_for_cur_split_df.to_csv(
            'results_%s.csv' % key,
            index=False)
        summary_df = pd.DataFrame([metrics_for_cur_split_df.median()])[[
            'mae_10', 'mae_50', 'mae_stat', 'mae_vel_inst', 'mae_vel_2sec']]
        summary_df.insert(0, 'split', key)
        print(summary_df.to_string(index=False))

        sum_list.append(summary_df)
    sum_df = pd.concat(sum_list)    
    print(sum_df.to_string(index=False))
