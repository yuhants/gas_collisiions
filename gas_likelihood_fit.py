import h5py
import sys

import numpy as np
from scipy.special import erf
from scipy.optimize import minimize

import calc_gas_collision_spectrum as gas

from multiprocessing import Pool

sphere = 'sphere_20260105'
length_search_window = 5e-5  # length of the search window (50 us)

eff_coefs = [1.07617456e+02, 7.92304675e-03] # 20260216: eff derived by counting pulses using timing
# eff_coefs = [2.88150446e+02, 1.08156304e-02]
eff_chi2  = 1
fit_band = (250, 400)
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

def calc_nll(sigma, drdqzn, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset=0):
    # Correct for signal efficiency (search and chi2 cut)
    hist_norm = np.sum(hist) * length_search_window * (bc[1] - bc[0])

    eff_qq = func_eff(bc, *eff_coefs)
    hist_gas = eff_chi2 * eff_qq * drdqzn * hist_norm

    idx_fit = np.logical_and(bc > fit_band[0], bc < fit_band[1])
    bi = bc[idx_fit]
    ni = hist[idx_fit]

    gas_contribution = np.sum(hist_gas[idx_fit])
    ntot = np.sum(ni) - gas_contribution

    if ntot > 0:
        # Use only the central value of pdf
        # faster and avoid numerical issues from integration
        # No correctiion for efficiency for background
        joint_pdf = gaus_normalized(bi, 0, sigma, fit_band)
        mui = ntot * joint_pdf * 50 + hist_gas[idx_fit]
    else:
        mui = hist_gas[idx_fit]
    mui[ mui < 1e-30 ] = 1e-30  # set a very small value so the log doesn't overflow

    return np.sum(np.nan_to_num(mui - ni * np.log(mui))) + nll_offset

def nll_sigma_pressure(sigma, log10_pressure, drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset=0):
    log10_pressure_ref = -8
    pressure_scale = np.power(10, (log10_pressure - log10_pressure_ref))

    _, drdqzn = gas.smear_drdqz_gauss(bc, drdqz, sigma)
    drdqzn_scaled = drdqzn * pressure_scale
    _nll = calc_nll(sigma, drdqzn_scaled, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset)

    return _nll

def minimize_nll(drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset):
    bounds = [(25, 125), (-13, -5)]
    args = [drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset]
    res = minimize(fun=lambda x: nll_sigma_pressure(*x, *args), x0=[70, -7],
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
        return np.nan, [np.nan, np.nan]
    
if __name__ == '__main__':
    # Read reconstruction data
    sphere = 'sphere_20260105'
    datasets_all = ['20260107_p8e_4e-8mbar', '20260107_p8e_3e-8mbar_valveclosed', 
                    '20260107_p8e_5e-8mbar', '20260107_p8e_6e-8mbar', '20260107_p8e_8e-8mbar', 
                    '20260107_p8e_1e-7mbar', '20260107_p8e_2e-7mbar']
    outfile_name = f'{sphere}_gas_recon_all.h5py'
    outdir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_recon'

    with h5py.File(f'{outdir}/{outfile_name}', 'r') as f:
        g = f['recon_histograms']
        bc = g['bc'][:]
        hh_0 = g[f'hist_{datasets_all[0]}'][:]
        hh_1 = g[f'hist_{datasets_all[1]}'][:]
        hh_2 = g[f'hist_{datasets_all[2]}'][:]
        hh_3 = g[f'hist_{datasets_all[3]}'][:]
        hh_4 = g[f'hist_{datasets_all[4]}'][:]
        hh_5 = g[f'hist_{datasets_all[5]}'][:]
        hh_6 = g[f'hist_{datasets_all[6]}'][:]

    # Read signal file
    signal_file = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_signal/xe_signal_5e-08nm_1e-08mbar.npz'
    with np.load(signal_file) as f:
        alpha_list = f['alpha_list']
        ts_list = f['ts_list']
        qq_kev = f['qq_kev']
        drdqzs = f['drdqz']

    out_dir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/likelihood_fit'
    hhs_all = [hh_0, hh_1, hh_2, hh_3, hh_4, hh_5, hh_6]
    for i, hist in enumerate(hhs_all):
        outfile = f'{datasets_all[i]}_likelihood_fit.npz'

        nlls_all = np.empty((alpha_list.size, ts_list.size))
        params_all = np.empty((alpha_list.size, ts_list.size, 2))
        for i in range(alpha_list.size):
            for j in range(ts_list.size):
                drdqz = drdqzs[i][j]
                _nll, _params = minimize_nll(drdqz, bc, hist, eff_coefs, eff_chi2, fit_band, nll_offset)
                nlls_all[i][j] = _nll
                params_all[i][j] = _params

        print(f'Writing file {out_dir}/{outfile}')
        np.savez(f'{out_dir}/{outfile}', alpha_list=alpha_list, ts_list=ts_list, nlls_all=nlls_all, params_all=params_all)