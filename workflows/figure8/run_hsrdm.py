import jax.numpy as jnp
import jax.scipy.stats as jstats
import numpy as np
from matplotlib import pyplot as plt
import os
import argparse
from collections import namedtuple

from dynagroup.io import ensure_dir
from dynagroup.params import dims_from_params
from dynagroup.model2a.gaussian.initialize import (
    PreInitialization_Strategy_For_CSP,
    smart_initialize_model_2a,
)
from dynagroup.vi.core import run_CAVI_with_JAX
from dynagroup.vi.M_step_and_ELBO import compute_elbo_decomposed
import dynagroup.vi.elbo_utils as elbo_utils

from dynagroup.util import (
    get_current_datetime_as_string,
    normalize_log_potentials_by_axis,
)
from dynagroup.model2a.figure8.diagnostics.fit_and_forecasting import (
    evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8,
)


# Load model setting specific to FigureEight
from setup_model import figure8_model_JAX, STP_prior

import setup_training as train_info
import setup_init as init_info

DEFAULT_RESULTS_DIR = os.path.join(
    os.environ.get("HOME"),
    'team-dynamics-time-series',
    'results',
    'figure8')


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True)
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir',
        default=os.environ.get(
            'RESULTS_DIR', DEFAULT_RESULTS_DIR),
        )
    parser.add_argument('--seed',
        default=1,
        type=int)
    parser.add_argument('--verbose',
        default=0,
        type=int)
    parser.add_argument('--use_provided_data_mask',
        default=1,
        type=int)
    parser.add_argument('--seeds_for_forecasting',
        type=str,
        default='120,121,122,123,124,125')
    # Allow user input to override default train_info vals
    train_info_keys = list()
    for key, val in train_info.__dict__.items():
        val_is_numeric = isinstance(val, (int, float, bool))
        if val_is_numeric:
            train_info_keys.append(key)
            parser.add_argument('--%s' % key,
                default=val,
                type=type(val))    
    # Allow user input to override default train_info vals
    init_info_keys = list()
    for key, val in init_info.__dict__.items():
        val_is_numeric = isinstance(val, (int, float, bool))
        if val_is_numeric:
            init_info_keys.append(key)
            parser.add_argument('--%s' % key,
                default=val,
                type=type(val))
    args = parser.parse_args()
    # Fill in train_info
    for key in train_info_keys:
        val = args.__dict__.get(key)
        if val != getattr(train_info, key):
            print("USER-PROVIDED OPTION: --%s %s" % (key, val))
            setattr(train_info, key, val)
    # Fill in init_info
    for key in init_info_keys:
        val = args.__dict__.get(key)
        if val != getattr(init_info, key):
            print("USER-PROVIDED OPTION: --%s %s" % (key, val))
            setattr(init_info, key, val)

    seed = args.seed
    datetime_as_string = get_current_datetime_as_string()
    use_provided_data_mask = args.use_provided_data_mask
    run_description = f"seed{seed:03d}_mask{use_provided_data_mask}_timestamp{datetime_as_string}"
    results_dir = os.path.join(
        args.results_dir, run_description) + os.path.sep
    ensure_dir(results_dir)

    # Load data itself (created on the fly)
    from create_data import (
        dataset,
        mask_TJ,
        TRUE_PARAMS,
        )
    T, J = mask_TJ.shape
    if not args.use_provided_data_mask:
        mask_TJ[:] = True
    DIMS = dims_from_params(TRUE_PARAMS)
    if dataset.example_end_times is None:
        dataset.example_end_times = np.array([-1, T])

    # dataset : contains observations as well as true s/z values
    #
    # Attributes
    # ----------
    # - .s : shape (T,)
    # - .zs : shape (T, J)
    # - .xs : shape (T, J, 2)
    print("dataset")
    print(" T=%d : num timesteps" % (dataset.xs.shape[0]))
    print(" D=%d : num obs dims" % (dataset.xs.shape[1]))
    print(" J=%d : num entities" % (DIMS.J))
    print("First 3 timesteps of first entity:")
    print(dataset.xs[:3, 0, :])
    print("Last 3 timesteps of last entity:")
    print(dataset.xs[-3:, -1, :])
    print("Num timesteps per entity hidden by mask:")
    print(T - mask_TJ.sum(axis=0)[:10])
    print("MODEL DIMS")
    print(" L=%d : num sys states\n K=%d : num entity states" % (
        DIMS.L, DIMS.K))


    # Initialize model
    init_result = smart_initialize_model_2a(
        DIMS,
        dataset.xs,
        dataset.example_end_times,
        figure8_model_JAX,
        init_info.strategy_for_CSP,
        init_info.init_n_em_iters_bottom,
        init_info.init_n_em_iters_top,
        seed,
        system_covariates=None,
        use_continuous_states=mask_TJ,
        save_dir=results_dir,
        verbose=args.verbose-1,
    )
    params_init = init_result.params
    VES_init, VEZ_init = init_result.ES_summary, init_result.EZ_summaries

    # params_init contains
    # - IP : initial tstep params
    # - STP : system transitions
    # - ETP : entity transitions
    # - EP : emissions params
    print("Params at Init")
    print("--------------")
    print("STP.Pi")
    print(params_init.STP.Pi[:3,:3])
    print("CSP.bs[j=0]")
    print(params_init.CSP.bs[0])
    print("CSP.bs[j=2]")
    print(params_init.CSP.bs[2])

    data_TJD = jnp.asarray(dataset.xs)

    VES, VEZ, params, ed_list, *_ = run_CAVI_with_JAX(
        params_init,
        VES_init, VEZ_init, STP_prior,
        figure8_model_JAX,
        data_TJD,
        dataset.example_end_times,
        train_info.n_cavi_iterations,
        train_info.Mstep_toggles,
        train_info.num_M_step_iters,
        system_covariates=None,
        use_continuous_states=mask_TJ,
        verbose=args.verbose,
        )

    print("Params after CAVI")
    print("--------------")
    print("STP.Pi")
    print(params.STP.Pi[:3,:3])
    print("CSP.bs[j=0]")
    print(params.CSP.bs[0])
    print("CSP.bs[j=2]")
    print(params.CSP.bs[2])

    elbo_obj = compute_elbo_decomposed(
        params,
        VES,
        VEZ,
        STP_prior,
        data_TJD,
        figure8_model_JAX,
        dataset.example_end_times,
        system_covariates=None,
    )
    n_tok = np.sum(mask_TJ) * DIMS.D
    elbo_pertok = elbo_obj.elbo / n_tok
    print("oldELBO after CAVI: ", "%.3f" % elbo_pertok)
    print("Entropy term   : ", "%.3f" % elbo_obj.entropy)

    elbo_dict = elbo_utils.calc_elbo(params, VES, VEZ, STP_prior,
        figure8_model_JAX,
        data_TJD, dataset.example_end_times, mask_TJ,
        return_dict=True)
    newelbo_pertok = float(elbo_dict['elbo']) / n_tok
    print("newELBO after CAVI: ", "%.3f" % newelbo_pertok)


    logp_pertok = jstats.norm.logpdf(data_TJD[mask_TJ], 0., 1.).sum() / n_tok
    print("baseline std normal log pdf:", "%.3f" % logp_pertok)

    import pandas as pd
    df = pd.DataFrame(ed_list)
    df['step'] = np.arange(df.shape[0])
    df[['step', 'elbo', 'energy', 'entropy', 'status']].to_csv(
        os.path.join(results_dir, 'info_per_step.csv'),
        index=False)

    ismissing_T = np.max(1-mask_TJ, axis=1)
    Tforecast = np.sum(ismissing_T)
    seeds_for_forecasting = None
    if len(args.seeds_for_forecasting) > 0:
        seeds_for_forecasting = np.asarray(
            [int(s) for s in args.seeds_for_forecasting.split(',')])
    if seeds_for_forecasting is not None and seeds_for_forecasting.size > 0:
        entity_idxs_for_forecasting = np.flatnonzero(
            np.sum(1-mask_TJ,axis=0)).tolist()
        _, _, _, forecasts = evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8(
            dataset.xs,
            params,
            VES,
            VEZ,
            Tforecast,
            figure8_model_JAX,
            seeds_for_forecasting,
            results_dir,
            use_continuous_states=mask_TJ,
            entity_idxs=entity_idxs_for_forecasting,
            filename_prefix="HSRDM",
            )
