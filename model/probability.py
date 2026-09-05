import numpy as np
from scipy.stats import norm

# E[|X|] = sigma * sqrt(2/pi) for X ~ Normal(0, sigma). Used to convert a
# mean-absolute-residual estimate into an equivalent standard deviation.
_MAE_TO_STD = np.sqrt(np.pi / 2)


def estimate_residual_scale(artifact, pred):
    """
    Returns the std-dev to use for a given point prediction `pred`, using the
    artifact's fitted variance model if present, else falling back to the
    single global residual_std saved by train.py.
    """
    global_std = artifact["residual_std"]
    variance_model = artifact.get("residual_std_model")

    if variance_model is None:
        return global_std

    pred_arr = np.atleast_1d(pred).astype(float).reshape(-1, 1)
    est_mae = variance_model.predict(pred_arr)
    est_std = np.clip(est_mae * _MAE_TO_STD, 0.3 * global_std, 3.0 * global_std)

    return est_std[0] if np.isscalar(pred) else est_std


def raw_p_over(pred, line, scale):
    """P(actual > line) assuming actual ~ Normal(pred, scale)."""
    return 1 - norm.cdf(line, loc=pred, scale=scale)


def apply_calibration(p_raw, calibration):
    """
    calibration: dict with 'method' in {'platt', 'isotonic', 'temperature'} and
    'model', as saved by calibrate.py. Pass None to skip calibration.
    """
    if calibration is None:
        return p_raw

    is_scalar = np.isscalar(p_raw)
    p_arr = np.clip(np.atleast_1d(p_raw).astype(float), 1e-6, 1 - 1e-6)
    method = calibration["method"]
    cal_model = calibration["model"]

    if method == "platt":
        p_cal = cal_model.predict_proba(p_arr.reshape(-1, 1))[:, 1]
    elif method == "isotonic":
        p_cal = cal_model.predict(p_arr)
    elif method == "temperature":
        T = cal_model
        logit = np.log(p_arr / (1 - p_arr))
        p_cal = 1 / (1 + np.exp(-logit / T))
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    return p_cal[0] if is_scalar else p_cal


def predict_p_over(artifact, pred, line, calibration=None):
    """
    Full pipeline: point prediction -> per-prediction residual scale ->
    P(over) via Normal CDF -> optional calibration correction.

    `artifact` is the dict loaded from a stat's .joblib file (needs at least
    'residual_std', optionally 'residual_std_model'). This is the one function
    that should be used anywhere a (prediction, line) pair needs to become a
    betting probability.
    """
    scale = estimate_residual_scale(artifact, pred)
    p_raw = raw_p_over(pred, line, scale)
    return apply_calibration(p_raw, calibration)