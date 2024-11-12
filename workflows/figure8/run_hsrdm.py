import jax.numpy as jnp
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

DEFAULT_RESULTS_DIR = os.path.join(
    os.environ.get("HOME"),
    'team-dynamics-time-series',
    'results',
    'figure8')


if __name__ == '__main__':
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
    parser.add_argument('--seeds_for_forecasting',
        type=str,
        default='120,121,122,123,124,125')
    args = parser.parse_args()
    seed = args.seed
    datetime_as_string = get_current_datetime_as_string()
    run_description = f"seed{seed:03d}_timestamp{datetime_as_string}"
    results_dir = os.path.join(
        args.results_dir, run_description) + os.path.sep
    ensure_dir(results_dir)

    # Load data itself (created on the fly)
    from create_data import (
        dataset,
        mask_TJ,
        TRUE_PARAMS,
        )
    DIMS = dims_from_params(TRUE_PARAMS)

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
    print("MODEL DIMS")
    print(" L=%d : num sys states\n K=%d : num entity states" % (
        DIMS.L, DIMS.K))

    InitArgs = namedtuple('InitArgs', [
        'seed', 
        'num_em_iters_top',
        'num_em_iters_bottom',
        'strategy_for_CSP'
        ])
    init_args = InitArgs(
        seed=1,
        num_em_iters_top=20,
        num_em_iters_bottom=5,
        strategy_for_CSP=PreInitialization_Strategy_For_CSP.LOCATION)

    # Initialize model
    init_result = smart_initialize_model_2a(
        DIMS,
        dataset.xs,
        dataset.example_end_times,
        figure8_model_JAX,
        init_args.strategy_for_CSP,
        init_args.num_em_iters_bottom,
        init_args.num_em_iters_top,
        init_args.seed,
        system_covariates=None,
        use_continuous_states=mask_TJ,
        save_dir=results_dir,
        verbose=args.verbose,
    )
    params_init = init_result.params
    # params_init contains
    # - IP : initial tstep params
    # - STP : system transitions
    # - ETP : entity transitions
    # - EP : emissions params

    VES, VEZ, params_learned, *_ = run_CAVI_with_JAX(
        jnp.asarray(dataset.xs),
        train_info.n_cavi_iterations,
        init_result,
        figure8_model_JAX,
        dataset.example_end_times,
        train_info.Mstep_toggles,
        train_info.num_M_step_iters,
        STP_prior,
        system_covariates=None,
        use_continuous_states=mask_TJ,
        verbose=args.verbose,
        )

    ismissing_T = np.max(1-mask_TJ, axis=1)
    Tforecast = np.sum(ismissing_T)
    entity_idxs_for_forecasting = np.flatnonzero(
        np.sum(1-mask_TJ,axis=0)).tolist()

    seeds_for_forecasting = np.asarray(
        [int(s) for s in args.seeds_for_forecasting.split(',')])
    _, _, _, forecasts = evaluate_and_plot_posterior_mean_and_forward_simulation_on_slice_for_figure_8(
        dataset.xs,
        params_learned,
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

