import warnings


def try_pomegranate_model_up_to_n_times(func, n: int):
    """
    Motivation:
        For some weird reason, the stupid pomegranate library will frequently (and randomly) error
        out when trying to learn a Gaussian mixture model.

        Ideally I would just use a more robust Gaussian mixture model.
        But I need for the GMM library to be able to take in sample weights.  Eventually perhaps just roll my own code.

        Note that the problem may not REALLY be the library; for instance some changepoint segments may have a mixture
        of clusters where there is a single point in the other cluster. I would need to handlethis case.
    """

    def wrapper(*args, **kwargs):
        result = None
        for i in range(n):
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                if (
                    "The factorization could not be completed because the input is not positive-definite"
                    in str(error)
                ) or ("'Normal' object has no attribute" in str(error)):
                    warnings.warn(
                        f"Error found in pomegrante function {func.__name__} on attempt {i+1}/{n}, rerunning."
                    )
                    continue
                else:
                    raise error
            if result is not None:
                break

        if result is None:
            warnings.warn(
                f"Could not identify non-None model for {func.__name__} even after {i+1}/{n} tries."
            )
        return result

    return wrapper


def try_changepoint_initialization_n_times(func, n: int):
    """
    Motivation:
        We are initializing the von Mises AR-HMM via a changepoint detector, but we might find fewer changepoints
        than specified regimes.  Ideally we'd actually use this info to help specify an entity-specific K. But for now
        we can at least try decreasing the penalty.
    """

    def wrapper(*args, **kwargs):
        CHANGEPOINT_PENALTY_DISCOUNT_FACTOR = 0.75
        result = None
        for i in range(n):
            try:
                result = func(*args, **kwargs)
            except Exception as error:
                if (
                    "Try reducing the changepoint penalty" in str(error)
                ) and "changepoint_penalty" in kwargs:
                    kwargs["changepoint_penalty"] *= CHANGEPOINT_PENALTY_DISCOUNT_FACTOR
                else:
                    raise error
            if result is not None:
                break
        if i > 0:
            warnings.warn(
                f"Needed to run vm AR-HMM via changepoint detection, {func.__name__} {i+1} times. "
            )
        return result

    return wrapper
