import h5py
import sys

import numpy as np
from scipy.special import erf, gammaln
from scipy.stats import chi2 as chi2_dist
from scipy.optimize import minimize

import calc_gas_collision_spectrum as gas

from multiprocessing import Pool

sphere = 'sphere_20260215'
length_search_window = 2**8 / 5e6  # 51.2 us — 2^8 samples at 5 MHz

# eff_coefs = [1.07617456e+02, 7.92304675e-03] # 20260216: eff derived by counting pulses using timing (sphere_20260105)
eff_coefs = [1.31615784e+02, 6.53825079e-03]  # for Sphere 20260215 with new reconstruction
eff_chi2  = 0.95
fit_band = (200, 500)
nll_offset = 0

def func_eff(x, z, f):
    return 0.5 * erf((x - z) * f) + 0.5

def gaus(x, mu, sigma):
    return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-1 * (x - mu)**2 / (2 * sigma**2))

def gaus_normalized(x, mu, sigma, fit_band):
    lb, ub = fit_band[0], fit_band[1]
    norm = 0.5 * (erf((ub - mu)/(np.sqrt(2)*sigma)) - erf((lb - mu)/(np.sqrt(2)*sigma)))

    x = np.asarray(x)
    if x.size == 1:
        return gaus(x, mu, sigma)[0] / norm
    else:
        return gaus(x, mu, sigma) / norm

def calc_nll(sigma, drdqzn, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset=0, signal_only=False):
    # Correct for signal efficiency (search and chi2 cut)
    hist_norm = np.sum(hist) * length_search_window * (bc[1] - bc[0])

    eff_qq = func_eff(bc, *eff_coefs)
    hist_gas = eff_chi2 * eff_qq * drdqzn * hist_norm

    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])
    bi = bc[idx_fit]
    ni = hist[idx_fit]

    if signal_only:
        mui = hist_gas[idx_fit].copy()
    else:
        gas_contribution = np.sum(hist_gas[idx_fit])
        ntot = np.sum(ni) - gas_contribution

        if ntot > 0:
            # Use only the central value of pdf
            # faster and avoid numerical issues from integration
            # No correctiion for efficiency for background
            joint_pdf = gaus_normalized(bi, 0, sigma, fit_band)
            mui = ntot * joint_pdf * (bc[1] - bc[0]) + hist_gas[idx_fit]
        else:
            mui = hist_gas[idx_fit]
    mui[ mui < 1e-30 ] = 1e-30  # set a very small value so the log doesn't overflow

    return np.sum(np.nan_to_num(mui - ni * np.log(mui))) + nll_offset

def negbin_logpmf(n, mu, k):
    """Negative binomial log-PMF.  E[n]=mu, Var[n]=mu+mu^2/k.  k->inf recovers Poisson."""
    mu = np.asarray(mu, dtype=np.float64)
    n  = np.asarray(n,  dtype=np.float64)
    mu = np.maximum(mu, 1e-30)
    p  = mu / (mu + k)
    return (gammaln(n + k) - gammaln(k) - gammaln(n + 1)
            + k * np.log1p(-p) + n * np.log(np.maximum(p, 1e-30)))

def calc_gof(mu, n, nparams, k=None, min_mu=1.0):
    """Pearson goodness-of-fit for a binned histogram.

    Parameters
    ----------
    mu      : (N,) expected counts per bin (model prediction)
    n       : (N,) observed counts per bin
    nparams : number of free parameters in the fit
    k       : negbin dispersion parameter.  If None, uses Poisson variance (var=mu).
    min_mu  : minimum expected counts for a bin to be included in the GOF sum.
              Bins with mu < min_mu are excluded (Pearson chi2 is unreliable there).

    Returns
    -------
    chi2_val : Pearson chi2 = sum((n-mu)^2 / var)
    ndof     : N - nparams
    chi2_ndf : chi2_val / ndof
    pvalue   : survival function P(X >= chi2_val) for X ~ chi2(ndof)
    """
    mu = np.asarray(mu, dtype=np.float64)
    n  = np.asarray(n,  dtype=np.float64)
    if k is not None:
        var = mu + mu**2 / k
    else:
        var = mu.copy()
    var = np.maximum(var, 1e-30)
    mask = mu >= min_mu
    chi2_val = np.sum((n[mask] - mu[mask])**2 / var[mask])
    ndof = max(int(np.sum(mask)) - nparams, 1)
    pvalue = chi2_dist.sf(chi2_val, ndof)
    return chi2_val, ndof, chi2_val / ndof, pvalue

def build_model_counts(sigma, drdqzn, bc, hist, eff_coefs, eff_chi2, fit_band, signal_only=False):
    """Build the expected counts per bin in the fit band.

    drdqzn should already include any amp_scale rescaling and pressure scaling.
    Returns (mu_i, idx_fit) where mu_i are the expected counts for bins in the fit band.
    """
    hist_norm = np.sum(hist) * length_search_window * (bc[1] - bc[0])
    eff_qq = func_eff(bc, *eff_coefs)
    hist_gas = eff_chi2 * eff_qq * drdqzn * hist_norm

    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])

    if signal_only:
        mui = hist_gas[idx_fit].copy()
    else:
        ni = hist[idx_fit]
        gas_contribution = np.sum(hist_gas[idx_fit])
        ntot = np.sum(ni) - gas_contribution

        if ntot > 0:
            joint_pdf = gaus_normalized(bc[idx_fit], 0, sigma, fit_band)
            mui = ntot * joint_pdf * (bc[1] - bc[0]) + hist_gas[idx_fit]
        else:
            mui = hist_gas[idx_fit].copy()
    mui = np.maximum(mui, 1e-30)
    return mui, idx_fit

def calc_nll_negbin(sigma, drdqzn, bc, hist, eff_coefs, eff_chi2, fit_band, k, nll_offset=0, signal_only=False):
    """NLL using negative binomial per bin (overdispersed Poisson)."""
    hist_norm = np.sum(hist) * length_search_window * (bc[1] - bc[0])

    eff_qq = func_eff(bc, *eff_coefs)
    hist_gas = eff_chi2 * eff_qq * drdqzn * hist_norm

    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])
    bi = bc[idx_fit]
    ni = hist[idx_fit]

    if signal_only:
        mui = hist_gas[idx_fit].copy()
    else:
        gas_contribution = np.sum(hist_gas[idx_fit])
        ntot = np.sum(ni) - gas_contribution

        # `mui` = expected signal + noise histogram in the fit band
        if ntot > 0:
            joint_pdf = gaus_normalized(bi, 0, sigma, fit_band)
            mui = ntot * joint_pdf * (bc[1] - bc[0]) + hist_gas[idx_fit]
        else:
            mui = hist_gas[idx_fit]
    mui = np.maximum(mui, 1e-30)

    return -np.sum(np.nan_to_num(negbin_logpmf(ni, mui, k))) + nll_offset

def nll_sigma_pressure_negbin(sigma, log10_pressure, drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, k, nll_offset=0, signal_only=False, amp_scale=1.0, amp_scale_sigma=None):
    log10_pressure_ref = -8
    pressure_scale = np.power(10, (log10_pressure - log10_pressure_ref))

    # Apply horizontal amplitude scale factor: measured = amp_scale * true
    if amp_scale != 1.0:
        drdqz_s = np.interp(bc / amp_scale, bc, drdqz) / amp_scale
    else:
        drdqz_s = drdqz

    _, drdqzn = gas.smear_drdqz_gauss(bc, drdqz_s, sigma)
    drdqzn_scaled = drdqzn * pressure_scale
    nll = calc_nll_negbin(sigma, drdqzn_scaled, bc, hist, eff_coefs, eff_chi2, fit_band, k, nll_offset, signal_only)

    # Gaussian constraint on amp_scale
    if amp_scale_sigma is not None:
        nll += 0.5 * ((amp_scale - 1.0) / amp_scale_sigma)**2

    return nll

def minimize_nll_negbin(drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, k, nll_offset, signal_only=False, amp_scale_sigma=None):
    """Minimize NLL with fixed overdispersion parameter k.

    If amp_scale_sigma is given, also fits an amplitude scale nuisance
    parameter with a Gaussian constraint of width amp_scale_sigma
    (e.g. 0.10 for 10 %).  Returns 3-element param vector
    [sigma, log10_p, amp_scale] in that case.
    """
    if amp_scale_sigma is not None:
        bounds = [(25, 125), (-13, -5), (0.7, 1.3)]
        def objective(x):
            return nll_sigma_pressure_negbin(
                x[0], x[1], drdqz, bc, hist, eff_coefs, eff_chi2, fit_band,
                k, nll_offset, signal_only, amp_scale=x[2], amp_scale_sigma=amp_scale_sigma)
        nan_result = (np.nan, [np.nan, np.nan, np.nan])

        # Coarse grid search to find good starting point
        sigma_grid = np.linspace(30, 100, 8)
        log10p_grid = np.linspace(-11, -5.5, 12)
        best_nll_grid = np.inf
        best_x0 = [60, -7, 1.0]
        for sg in sigma_grid:
            for pg in log10p_grid:
                val = objective([sg, pg, 1.0])
                if val < best_nll_grid:
                    best_nll_grid = val
                    best_x0 = [sg, pg, 1.0]
    else:
        bounds = [(25, 125), (-13, -5)]
        def objective(x):
            return nll_sigma_pressure_negbin(
                x[0], x[1], drdqz, bc, hist, eff_coefs, eff_chi2, fit_band,
                k, nll_offset, signal_only)
        nan_result = (np.nan, [np.nan, np.nan])

        sigma_grid = np.linspace(30, 100, 8)
        log10p_grid = np.linspace(-11, -5.5, 12)
        best_nll_grid = np.inf
        best_x0 = [60, -7]
        for sg in sigma_grid:
            for pg in log10p_grid:
                val = objective([sg, pg])
                if val < best_nll_grid:
                    best_nll_grid = val
                    best_x0 = [sg, pg]

    res = minimize(fun=objective, x0=best_x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'disp' : False,
                            'maxiter': 50000,
                            'maxfev': 50000,
                            'adaptive': True,
                            'fatol': 0.001,
                            }
                    )
    if res.success:
        return res.fun, res.x
    else:
        return nan_result

def make_drdqz_interpolator(signal_file):
    """Build a RegularGridInterpolator for drdqz over (alpha, T_sensor).

    Parameters
    ----------
    signal_file : str
        Path to .npz file with keys: alpha_list, ts_list, qq_kev, drdqz.
        drdqz has shape (n_alpha, n_ts, n_qq).

    Returns
    -------
    interp : RegularGridInterpolator
        Callable as interp((alpha, T_sensor)) → drdqz array of shape (n_qq,).
    qq_kev : array
        The momentum-transfer grid.
    alpha_bounds : (float, float)
    ts_bounds : (float, float)
    """
    from scipy.interpolate import RegularGridInterpolator
    with np.load(signal_file) as f:
        alpha_list = f['alpha_list']
        ts_list = f['ts_list']
        qq_kev = f['qq_kev']
        drdqzs = f['drdqz']  # (n_alpha, n_ts, n_qq)
    interp = RegularGridInterpolator(
        (alpha_list, ts_list), drdqzs,
        method='linear', bounds_error=True,
    )
    return interp, qq_kev, (alpha_list[0], alpha_list[-1]), (ts_list[0], ts_list[-1])


def nll_full(sigma, log10_pressure, alpha, T_sensor,
             drdqz_interp, qq_kev, bc, hist,
             eff_coefs, eff_chi2, fit_band, k,
             nll_offset=0, signal_only=False,
             amp_scale=1.0, amp_scale_sigma=None):
    """NLL with alpha and T_sensor as free parameters.

    Interpolates drdqz from precomputed grid, then delegates to
    nll_sigma_pressure_negbin (or Poisson if k is None).
    """
    drdqz_raw = drdqz_interp((alpha, T_sensor))       # shape (n_qq,)
    drdqz_bc = np.interp(bc, qq_kev, drdqz_raw)       # onto histogram bins

    if k is not None:
        return nll_sigma_pressure_negbin(
            sigma, log10_pressure, drdqz_bc, bc, hist,
            eff_coefs, eff_chi2, fit_band, k, nll_offset,
            signal_only, amp_scale, amp_scale_sigma)
    else:
        return nll_sigma_pressure(
            sigma, log10_pressure, drdqz_bc, bc, hist,
            eff_coefs, eff_chi2, fit_band, nll_offset,
            signal_only, amp_scale, amp_scale_sigma)


def minimize_nll_full(drdqz_interp, qq_kev, bc, hist,
                      eff_coefs, eff_chi2, fit_band,
                      k=None, nll_offset=0, signal_only=False,
                      amp_scale_sigma=None,
                      alpha_bounds=(0.0, 1.0), ts_bounds=(294.0, 1000.0)):
    """Minimize NLL over (sigma, log10_p, alpha, T_sensor).

    Optionally also fits amp_scale if amp_scale_sigma is given (5-param fit).

    Parameters
    ----------
    drdqz_interp : RegularGridInterpolator
        From make_drdqz_interpolator().
    qq_kev : array
        Momentum-transfer grid matching the interpolator.
    alpha_bounds, ts_bounds : (lo, hi)
        Bounds for alpha and T_sensor.

    Returns
    -------
    nll_min : float
    params  : array — [sigma, log10_p, alpha, T_sensor] or
              [sigma, log10_p, alpha, T_sensor, amp_scale]
    """
    has_amp = amp_scale_sigma is not None

    if has_amp:
        bounds = [(25, 125), (-13, -5), alpha_bounds, ts_bounds, (0.7, 1.3)]
        def objective(x):
            return nll_full(x[0], x[1], x[2], x[3],
                            drdqz_interp, qq_kev, bc, hist,
                            eff_coefs, eff_chi2, fit_band, k, nll_offset,
                            signal_only, amp_scale=x[4],
                            amp_scale_sigma=amp_scale_sigma)
        nan_result = (np.nan, np.full(5, np.nan))
    else:
        bounds = [(25, 125), (-13, -5), alpha_bounds, ts_bounds]
        def objective(x):
            return nll_full(x[0], x[1], x[2], x[3],
                            drdqz_interp, qq_kev, bc, hist,
                            eff_coefs, eff_chi2, fit_band, k, nll_offset,
                            signal_only)
        nan_result = (np.nan, np.full(4, np.nan))

    # Coarse grid search for starting point
    sigma_grid = np.linspace(40, 100, 7)
    log10p_grid = np.linspace(-11, -5.5, 8)
    alpha_grid = np.linspace(alpha_bounds[0], alpha_bounds[1], 6)
    ts_grid = np.linspace(ts_bounds[0], ts_bounds[1], 6)

    best_nll_grid = np.inf
    best_x0 = [60, -7, 0.5, 500] + ([1.0] if has_amp else [])
    for sg in sigma_grid:
        for pg in log10p_grid:
            for ag in alpha_grid:
                for tg in ts_grid:
                    x = [sg, pg, ag, tg] + ([1.0] if has_amp else [])
                    val = objective(x)
                    if val < best_nll_grid:
                        best_nll_grid = val
                        best_x0 = list(x)

    res = minimize(fun=objective, x0=best_x0,
                   method='Nelder-Mead',
                   bounds=bounds,
                   options={'disp': False,
                            'maxiter': 100000,
                            'maxfev': 100000,
                            'adaptive': True,
                            'fatol': 0.001})
    if res.success:
        return res.fun, res.x
    else:
        return nan_result


def profile_log10_p_negbin(drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, k,
                           nll_offset=0, signal_only=False, amp_scale_sigma=None,
                           best_params=None, delta_nll=0.5, n_scan=60, scan_half_width=1.5):
    """Profile likelihood scan over log10_pressure to get 1-sigma errors.

    Parameters
    ----------
    best_params : array-like [sigma, log10_p, amp_scale] or [sigma, log10_p]
        Best-fit parameters from minimize_nll_negbin.
    delta_nll : float
        NLL increase defining the confidence interval (0.5 = 1-sigma).
    n_scan : int
        Number of log10_p points to scan on each side of the best fit.
    scan_half_width : float
        Half-width of the scan range in log10_p units.

    Returns
    -------
    log10_p_lo, log10_p_hi : float
        Lower and upper 1-sigma bounds on log10_pressure.
    err_lo, err_hi : float
        Asymmetric errors: best - lo, hi - best (both positive).
    profile_log10_p : array
        Scanned log10_p values.
    profile_nll : array
        Profiled NLL values.
    """
    if best_params is None:
        raise ValueError('best_params must be provided')

    has_amp_scale = amp_scale_sigma is not None
    sigma_bf = best_params[0]
    log10_p_bf = best_params[1]
    amp_scale_bf = best_params[2] if has_amp_scale else 1.0

    # Scan range
    log10_p_lo_scan = max(log10_p_bf - scan_half_width, -13)
    log10_p_hi_scan = min(log10_p_bf + scan_half_width, -5)
    log10_p_scan = np.linspace(log10_p_lo_scan, log10_p_hi_scan, 2 * n_scan + 1)

    profile_nll = np.empty_like(log10_p_scan)

    for i, log10_p in enumerate(log10_p_scan):
        if has_amp_scale:
            def obj(x):
                return nll_sigma_pressure_negbin(
                    x[0], log10_p, drdqz, bc, hist, eff_coefs, eff_chi2,
                    fit_band, k, nll_offset, signal_only,
                    amp_scale=x[1], amp_scale_sigma=amp_scale_sigma)
            x0 = [sigma_bf, amp_scale_bf]
            bounds = [(25, 125), (0.7, 1.3)]
        else:
            def obj(x):
                return nll_sigma_pressure_negbin(
                    x[0], log10_p, drdqz, bc, hist, eff_coefs, eff_chi2,
                    fit_band, k, nll_offset, signal_only)
            x0 = [sigma_bf]
            bounds = [(25, 125)]

        res = minimize(obj, x0, method='Nelder-Mead', bounds=bounds,
                       options={'maxiter': 10000, 'fatol': 0.001, 'adaptive': True})
        profile_nll[i] = res.fun if res.success else np.nan

    nll_min = np.nanmin(profile_nll)
    dnll = profile_nll - nll_min

    # Find crossings at delta_nll
    idx_min = np.nanargmin(dnll)
    log10_p_lo_bound = np.nan
    log10_p_hi_bound = np.nan

    # Left side: walk left from minimum, find where dnll crosses delta_nll upward
    for j in range(idx_min, 0, -1):
        if not (np.isnan(dnll[j-1]) or np.isnan(dnll[j])):
            if dnll[j] < delta_nll and dnll[j-1] >= delta_nll:
                frac = (delta_nll - dnll[j]) / (dnll[j-1] - dnll[j])
                log10_p_lo_bound = log10_p_scan[j] + frac * (log10_p_scan[j-1] - log10_p_scan[j])
                break

    # Right side: walk right from minimum, find where dnll crosses delta_nll upward
    for j in range(idx_min, len(dnll) - 1):
        if not (np.isnan(dnll[j]) or np.isnan(dnll[j+1])):
            if dnll[j] < delta_nll and dnll[j+1] >= delta_nll:
                frac = (delta_nll - dnll[j]) / (dnll[j+1] - dnll[j])
                log10_p_hi_bound = log10_p_scan[j] + frac * (log10_p_scan[j+1] - log10_p_scan[j])
                break

    err_lo = log10_p_bf - log10_p_lo_bound  # positive
    err_hi = log10_p_hi_bound - log10_p_bf  # positive

    return log10_p_lo_bound, log10_p_hi_bound, err_lo, err_hi, log10_p_scan, profile_nll


def nll_sigma_pressure(sigma, log10_pressure, drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset=0, signal_only=False, amp_scale=1.0, amp_scale_sigma=None):
    log10_pressure_ref = -8
    pressure_scale = np.power(10, (log10_pressure - log10_pressure_ref))

    if amp_scale != 1.0:
        drdqz_s = np.interp(bc / amp_scale, bc, drdqz) / amp_scale
    else:
        drdqz_s = drdqz

    _, drdqzn = gas.smear_drdqz_gauss(bc, drdqz_s, sigma)
    drdqzn_scaled = drdqzn * pressure_scale
    _nll = calc_nll(sigma, drdqzn_scaled, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset, signal_only)

    if amp_scale_sigma is not None:
        _nll += 0.5 * ((amp_scale - 1.0) / amp_scale_sigma)**2

    return _nll

def minimize_nll(drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset, signal_only=False, amp_scale_sigma=None):
    if amp_scale_sigma is not None:
        bounds = [(25, 125), (-13, -5), (0.7, 1.3)]
        def objective(x):
            return nll_sigma_pressure(
                x[0], x[1], drdqz, bc, hist, eff_coefs, eff_chi2, fit_band,
                nll_offset, signal_only, amp_scale=x[2], amp_scale_sigma=amp_scale_sigma)
        nan_result = (np.nan, [np.nan, np.nan, np.nan])

        sigma_grid = np.linspace(30, 100, 8)
        log10p_grid = np.linspace(-11, -5.5, 12)
        best_nll_grid = np.inf
        best_x0 = [70, -7, 1.0]
        for sg in sigma_grid:
            for pg in log10p_grid:
                val = objective([sg, pg, 1.0])
                if val < best_nll_grid:
                    best_nll_grid = val
                    best_x0 = [sg, pg, 1.0]
    else:
        bounds = [(25, 125), (-13, -5)]
        def objective(x):
            return nll_sigma_pressure(
                x[0], x[1], drdqz, bc, hist, eff_coefs, eff_chi2, fit_band,
                nll_offset, signal_only)
        nan_result = (np.nan, [np.nan, np.nan])

        sigma_grid = np.linspace(30, 100, 8)
        log10p_grid = np.linspace(-11, -5.5, 12)
        best_nll_grid = np.inf
        best_x0 = [70, -7]
        for sg in sigma_grid:
            for pg in log10p_grid:
                val = objective([sg, pg])
                if val < best_nll_grid:
                    best_nll_grid = val
                    best_x0 = [sg, pg]

    res = minimize(fun=objective, x0=best_x0,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'disp' : False,
                            'maxiter': 50000,
                            'maxfev': 50000,
                            'adaptive': True,
                            'fatol': 0.001,
                            }
                    )
    if res.success:
        return res.fun, res.x
    else:
        return nan_result
    
def calc_nll_mc(sigma, mc_signal, bc, hist, fit_band, nll_offset=0):
    """NLL using MC signal template (efficiency/chi2 cuts already applied in MC).

    Parameters
    ----------
    sigma      : background Gaussian width (keV/c)
    mc_signal  : MC signal histogram (counts), same binning as hist
    bc         : bin centres (keV/c)
    hist       : observed data histogram (counts)
    fit_band   : (lo, hi) keV/c
    """
    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])
    ni = hist[idx_fit]
    hist_gas = mc_signal[idx_fit].astype(np.float64)

    gas_contribution = np.sum(hist_gas)
    ntot = np.sum(ni) - gas_contribution

    if ntot > 0:
        joint_pdf = gaus_normalized(bc[idx_fit], 0, sigma, fit_band)
        mui = ntot * joint_pdf * (bc[1] - bc[0]) + hist_gas
    else:
        mui = hist_gas.copy()
    mui = np.maximum(mui, 1e-30)

    return np.sum(np.nan_to_num(mui - ni * np.log(mui))) + nll_offset

def calc_nll_mc_negbin(sigma, mc_signal, bc, hist, fit_band, k, nll_offset=0):
    """Negative-binomial NLL using MC signal template."""
    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])
    ni = hist[idx_fit]
    hist_gas = mc_signal[idx_fit].astype(np.float64)

    gas_contribution = np.sum(hist_gas)
    ntot = np.sum(ni) - gas_contribution

    if ntot > 0:
        joint_pdf = gaus_normalized(bc[idx_fit], 0, sigma, fit_band)
        mui = ntot * joint_pdf * (bc[1] - bc[0]) + hist_gas
    else:
        mui = hist_gas.copy()
    mui = np.maximum(mui, 1e-30)

    return -np.sum(np.nan_to_num(negbin_logpmf(ni, mui, k))) + nll_offset

def minimize_nll_mc(mc_signal, bc, hist, fit_band, nll_offset, k=None):
    """Minimize NLL over sigma only, using MC signal template.

    If k is given, uses negative-binomial likelihood; otherwise Poisson.
    """
    if k is not None:
        fun = lambda x: calc_nll_mc_negbin(x[0], mc_signal, bc, hist, fit_band, k, nll_offset)
    else:
        fun = lambda x: calc_nll_mc(x[0], mc_signal, bc, hist, fit_band, nll_offset)

    res = minimize(fun=fun, x0=[70],
                    method='Nelder-Mead',
                    bounds=[(25, 200)],
                    options={'disp': False,
                            'maxiter': 50000,
                            'maxfev': 50000,
                            'adaptive': True,
                            'fatol': 0.001,
                            }
                    )
    if res.success:
        return res.fun, res.x
    else:
        return np.nan, [np.nan]


def load_data_hists(recon_file, gas='xenon', hist_suffix='_chi2_nocal'):
    """Load histograms from a real-data recon file.

    Returns (bc, data_hists) where data_hists is an OrderedDict
    {dataset_name: {'hist': array, 'pressure_mbar': float}},
    sorted by pressure ascending.
    """
    _stage_suffixes = (
        '_nocal', '_cal',
        '_nocuts', '_nocuts_nocal', '_nocuts_cal',
        '_hom_bal', '_hom_bal_nocal', '_hom_bal_cal',
        '_noise', '_noise_nocal', '_noise_cal',
        '_drive', '_drive_nocal', '_drive_cal',
        '_chi2', '_chi2_nocal', '_chi2_cal',
    )
    from collections import OrderedDict
    with h5py.File(recon_file, 'r') as f:
        g = f['recon_histograms']
        bc = g['bc'][:]
        grp = g[gas]
        datasets = [k for k in grp.keys()
                    if not any(k.endswith(s) for s in _stage_suffixes)]
        superseded = {d[:-2] for d in datasets if d.endswith('_1')}
        datasets = sorted([d for d in datasets if d not in superseded],
                          key=lambda d: grp[d].attrs['pressure_mbar'])
        data_hists = OrderedDict()
        for d in datasets:
            key = d + hist_suffix
            data_hists[d] = {
                'hist': grp[key][:],
                'pressure_mbar': grp[key].attrs['pressure_mbar'],
            }
    return bc, data_hists

def load_mc_hists(mc_file, chi2_thr=600):
    """Load MC signal templates from mc_xe_chi2_scan.h5py.

    Returns (bc, mc_pressures, mc_meta) where mc_pressures is
    {pressure_mbar: histogram_array} and mc_meta is a dict of MC parameters.
    """
    mc_chi2_key = f'chi2_{chi2_thr}'
    with h5py.File(mc_file, 'r') as f:
        mc_meta = dict(f.attrs)
        mc_grp = f[f'recon_histograms/{mc_chi2_key}']
        bc = f['recon_histograms/bc'][:]
        mc_pressures = {}
        for pkey in mc_grp.keys():
            p = mc_grp[pkey].attrs['pressure_mbar']
            mc_pressures[p] = mc_grp[pkey][:]
    return bc, mc_pressures, mc_meta


if __name__ == '__main__':
    import os

    # ── Configuration ─────────────────────────────────────────────────────
    sphere = 'sphere_20260215'
    k_negbin = None                       # set to a float to use negbin likelihood

    base_dir = r'/Users/yuhan/work/nanospheres/gas_collisions/data_processed'
    recon_file = os.path.join(base_dir, 'gas_recon', f'{sphere}_gas_recon_all.h5py')
    signal_file = os.path.join(base_dir, 'gas_signal', 'xe_signal_5e-08nm_1e-08mbar.npz')
    out_dir    = os.path.join(base_dir, 'likelihood_fit')
    os.makedirs(out_dir, exist_ok=True)

    # ── Load data histograms ──────────────────────────────────────────────
    bc, data_hists = load_data_hists(recon_file, gas='xenon', hist_suffix='_chi2_nocal')
    print(f'Loaded {len(data_hists)} xenon datasets from {recon_file}')
    for d, info in data_hists.items():
        print(f'  {d}  p = {info["pressure_mbar"]:.2e} mbar')

    # ── Load analytic signal model ────────────────────────────────────────
    with np.load(signal_file) as f:
        alpha_list = f['alpha_list']
        ts_list = f['ts_list']
        drdqzs = f['drdqz']
    print(f'\nSignal model: {signal_file}')
    print(f'  alpha: {alpha_list.size} points [{alpha_list[0]:.3f}, {alpha_list[-1]:.3f}]')
    print(f'  T_sensor: {ts_list.size} points [{ts_list[0]:.1f}, {ts_list[-1]:.1f}] K')

    # ── Fit: alpha x T_sensor scan ────────────────────────────────────────
    for d, info in data_hists.items():
        outfile = os.path.join(out_dir, f'{d}_likelihood_fit.npz')

        nlls_all = np.empty((alpha_list.size, ts_list.size))
        params_all = np.empty((alpha_list.size, ts_list.size, 2))
        for i in range(alpha_list.size):
            for j in range(ts_list.size):
                drdqz = drdqzs[i][j]
                if k_negbin is not None:
                    _nll, _params = minimize_nll_negbin(drdqz, bc, info['hist'],
                                                        eff_coefs, eff_chi2, fit_band,
                                                        k_negbin, nll_offset)
                else:
                    _nll, _params = minimize_nll(drdqz, bc, info['hist'],
                                                  eff_coefs, eff_chi2, fit_band, nll_offset)
                nlls_all[i][j] = _nll
                params_all[i][j] = _params

        print(f'Writing {outfile}')
        np.savez(outfile, alpha_list=alpha_list, ts_list=ts_list,
                 nlls_all=nlls_all, params_all=params_all,
                 dataset=d, data_pressure_mbar=info['pressure_mbar'],
                 k_negbin=k_negbin if k_negbin is not None else -1)