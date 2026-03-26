import numpy as np
import os
import h5py

from scipy.signal import decimate
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import analysis_utils as utils

sphere = 'sphere_20260215'

# ============================================================
# CONFIG — edit these before running
# ============================================================

# Impulse reconstruction filter setting
# bandpass_lb, bandpass_ub = (35000, 70000)  # sphere_20251212
# bandpass_lb, bandpass_ub = (39000, 74000)  # sphere_20260105
# bandpass_lb, bandpass_ub = (38000, 75000)  # sphere_20260215
bandpass_lb, bandpass_ub = (35000, 80000)   # Analysis bandwidth in Hz (Sphere 20260215; after introducing imprecision)
lowpass_order = 3
notch_freq = 137000

# Fixed reconstruction parameters. Set to a float to pin to a constant for all
# datasets/files instead of using the per-file Voigt fit result.
# Set to None to use the per-file fit (default behaviour).
fixed_gamma_damping = 1 * 2 * np.pi   # rad/s, e.g. 200 * 2 * np.pi
fixed_c_imp         = 1.5e-22   # raw units (normalized to susceptibility); bypasses c_imp_voigt × c_imp_scaling entirely

# An arbitrary scaling parameter for the noise floor in
# the noise model
# use a a fallback when c_imp_scan failed
c_imp_scaling = 1/5

# Per-dataset c_imp_scaling overrides for non-CAL_MODE processing.
# Keyed by (data_type, dataset_name) — same values as in datasets_config.
# Takes priority over the value stored in the cal summary HDF5.
# Use when the scan-optimised scaling produces too much ringing for a specific dataset.
c_imp_scaling_overrides = {
    ('background_data', '20260219_p6e_3e-8mbar_sf6valveclosed'): 1/5,
    ('krypton_data', '20260219_p6e_7e-7mbar'): 1/5,
    ('sf6_data', '20260219_p6e_1e-6mbar'): 1/5,
    ('sf6_data', '20260219_p6e_7e-8mbar'): 1/5,
}

# Params for identifying pulse indices
positive_pulse = True
trigger_val = 0.5 * positive_pulse

analysis_window_length_gas = 2**19    # Length of analysis window in number of indices

fit_window_length      = 2**19   # Window length to fit for frequencies
analysis_window_length = 2**18   # Length of analysis window in number of indices
search_window_length   = 2**8     # 50 us search window

# If True, process each triggered pulse with recon_pulse() (same as
# process_impulse_calibration.py).  All files in a dataset are combined
# into a single output HDF5 with per-pulse 1-D arrays.
CAL_MODE_DEFAULT = False

# If True (requires CAL_MODE=True), scan over c_imp_scaling values using the
# first scan_n_files files, select the one with the best amplitude resolution
# (minimum Gaussian sigma/mean), then process all files with that scaling.
# Saves a plot and records the chosen scaling as an HDF5 attribute.
SCAN_C_IMP = False
c_imp_scan_values = [1/50, 1/30, 1/10, 1/5, 1/3, 1, 3]   # scaling factors to scan
scan_n_files      = 5                           # pilot files per dataset (good files)
scan_max_load     = 20                          # max files to screen for outliers (>= scan_n_files)
outlier_c_imp_factor = 3.0                      # reject pilot files with c_imp_voigt > factor × median
sigma_kev_threshold  = 150                      # keV — fall back to default c_imp_scaling if best σ exceeds this
expected_kev      = 1046                        # expected cal pulse amplitude in keV/c (for plot)
waveform_half_len = 1500                        # samples each side of peak to store (~300 µs at 5 MHz)

# Amplitude to keV calibration factor (from calibration dataset; used as fallback
# in scan plot if Gaussian fit fails for a given scaling)
# amp2kev_from_cal = 6792.86423779262    # sphere_20260105
# amp2kev_from_cal = 8363.560351624732   # sphere_20260215
amp2kev_from_cal = 10497.219118653622        # sphere_20260215; after introducing imprecision

# For calculating chi2, we simply assume an approximate 60 keV sigma
# (not currently used — process_dataset() derives ds_sigma_amp from per-dataset amp2kev)
# sigma_p_amp = 60 / amp2kev_from_cal

# Base directory for per-dataset calibration summaries (produced by gas_recon notebook).
# Used in non-CAL_MODE to load per-dataset c_imp_scaling, amp2kev, and pulse template.
# The summary for each gas type lives at:
#   {cal_summary_base}/{data_type}/{sphere}_{gas_type}_cal_summary.hdf5
# where gas_type = data_type with the trailing '_data' stripped.
cal_summary_base  = (rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/'
                     rf'gas_data_processed/{sphere}')
template_half_len = 250   # samples each side of peak used as chi-square template (500 total)

# Datasets to process, grouped by gas type.
# Each entry: (dataset_name, data_type_folder, file_prefix, n_files)
datasets_config = {
    'background': [
        ('20260219_p6e_4e-8mbar',                'background_data', '20260219_dfg_', 25),
        ('20260219_p6e_3e-8mbar_xevalveclosed',  'background_data', '20260219_dfg_', 25),
        ('20260219_p6e_3e-8mbar_krvalveclosed',  'background_data', '20260219_dfg_', 25),
        ('20260219_p6e_3e-8mbar_sf6valveclosed', 'background_data', '20260219_dfg_', 25),
    ],
    'xenon': [
        ('20260219_p6e_1e-6mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_8e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        # # ('20260219_p6e_4e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_4e-7mbar_1', 'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_2e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        # # ('20260219_p6e_1e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar_1', 'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-8mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-8mbar',   'xenon_data', '20260219_dfg_', 25),
    ],
    'krypton': [
        ('20260219_p6e_1e-6mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-7mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-7mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_2e-7mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-8mbar', 'krypton_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-8mbar', 'krypton_data', '20260219_dfg_', 25),
    ],
    'sf6': [
        ('20260219_p6e_1e-6mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        # ('20260219_p6e_3e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_3e-7mbar_1', 'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-8mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-8mbar',   'sf6_data', '20260219_dfg_', 25),
    ],
}
# ============================================================

def get_idx_in_window(amp_searched_idx, search_window_length, lb):
    ret = np.empty_like(amp_searched_idx)

    for i, amp_idx in enumerate(amp_searched_idx):
        ret[i] = amp_idx + lb + search_window_length * i
    
    return ret

def homodyne_lose_lock(zz_windowed, zz_bp_windowed):
    # Z signal out of balance, meaning that homodyne losing lock
    if np.abs(np.mean(zz_windowed)) > 0.5:
        return True
    
    if np.max(np.abs(zz_windowed)) > 0.95:
        return True

    # Check the sum over 100 indices to see if there
    # is a consecutive period of very small signal after bandpass
    convolved = np.convolve(np.abs(zz_bp_windowed),np.ones(100, dtype=int), 'valid')
    if np.sum(convolved < 1e-3) > 0:
        return True

def get_normalized_template(sphere, bounds=(1250, 1750), downsampled=False):
    pulse_shape_file = np.load(rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/pulse_calibration/{sphere}/{sphere}_impulse_recon_combined.npz')
    pulse_shape_template = pulse_shape_file['ps_20v']

    normalized_template = pulse_shape_template / np.max(pulse_shape_template)

     # Take the central values around the peak
    ret = normalized_template[bounds[0]:bounds[1]]

    # Downsample to 500 kHz (so the 200 us template has 100 indices)
    if downsampled:
        ret_downsampled = decimate(ret, 10)
        return ret_downsampled / np.max(ret_downsampled)
    else:
        return ret

def load_cal_from_npz(sphere, bounds=(1250, 1750)):
    """Load 20V pulse template and amp2kev directly from the calibration npz.

    Used when both fixed_gamma_damping and fixed_c_imp are set, bypassing
    the per-dataset cal summary HDF5 entirely.
    """
    npz_path = rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/pulse_calibration/{sphere}/{sphere}_impulse_recon_combined.npz'
    pulse_shape_file = np.load(npz_path)
    amp2kev = float(pulse_shape_file['amp2kev'])

    template = pulse_shape_file['ps_20v']
    template = template / np.max(template)
    template = template[bounds[0]:bounds[1]]

    return template, amp2kev

def load_dataset_cal(dataset, data_type):
    """Load per-dataset calibration from the gas-type cal summary HDF5.

    The summary file is resolved as:
      {cal_summary_base}/{data_type}/{sphere}_{gas_type}_cal_summary.hdf5
    where gas_type = data_type with '_data' stripped (e.g. 'xenon_data' → 'xenon').

    Returns (template, ds_c_imp_scaling, ds_amp2kev).
    Falls back to (default_template, c_imp_scaling, amp2kev_from_cal) if the
    dataset is not found in the summary or the file cannot be opened.
    """
    gas_type = data_type.removesuffix('_data')
    cal_summary_path = os.path.join(cal_summary_base, data_type,
                                    f'{sphere}_{gas_type}_cal_summary.hdf5')
    default_template = get_normalized_template(sphere, bounds=(1250, 1750), downsampled=False)
    try:
        with h5py.File(cal_summary_path, 'r') as f:
            if dataset not in f:
                print(f'  [{dataset}] not in cal summary — using defaults')
                return default_template, c_imp_scaling, amp2kev_from_cal
            g = f[dataset]
            ds_c_imp_scaling = float(g.attrs['c_imp_scaling'])
            ds_amp2kev       = float(g['amp2kev'][()])
            norm_wav         = g['norm_waveform'][:]
        peak     = len(norm_wav) // 2
        template = norm_wav[peak - template_half_len : peak + template_half_len]
        override = c_imp_scaling_overrides.get((data_type, dataset))
        if override is not None:
            print(f'  [{dataset}] c_imp_scaling override: {override:.4g} (HDF5 had {ds_c_imp_scaling:.4g}), amp2kev={ds_amp2kev:.0f}')
            ds_c_imp_scaling = override
        else:
            print(f'  [{dataset}] cal summary: c_imp_scaling={ds_c_imp_scaling:.4g}, amp2kev={ds_amp2kev:.0f}')
        return template, ds_c_imp_scaling, ds_amp2kev
    except Exception as e:
        print(f'  [{dataset}] cal summary load failed ({e}) — using defaults')
        return default_template, c_imp_scaling, amp2kev_from_cal


def calc_chisquares(amp_lp, indices_in_window, normalized_template, sigma_amp):
    ret = np.empty(indices_in_window.shape, np.float64)

    window_size = int(normalized_template.size / 2)
    for i, idx in enumerate(indices_in_window):
        amp = amp_lp[idx]
        waveform = amp_lp[idx-window_size : idx+window_size]

        # Amplitude can be negative so no need to adjust for polarity
        template_scaled = amp * normalized_template

        # Sigma should be in amplitude (not keV)
        ret[i] = np.sum( ((waveform - template_scaled)/sigma_amp)**2 )
    return ret

def get_driven_power(dt, zz_windowed, drive_freq):
    ff, pp = utils.get_psd(dt=dt, zz=zz_windowed, nperseg=2**16)
    noise_idx = np.logical_and(ff > 150000, ff < 175000)
    noise_floor = np.mean(pp[noise_idx])

    search_idx = np.logical_and(ff > 30000, ff < 60000)
    f_res = ff[search_idx][np.argmax(pp[search_idx])]

    drive_area = utils.get_area_driven_peak(ff, pp, passband=(drive_freq-100, drive_freq+100), noise_floor=noise_floor, plot=False)
    return f_res, drive_area

def process_dataset(sphere, dataset, type, data_prefix, nfile, idx_start):
    data_dir = rf'/Volumes/LaCie/gas_collisions/{type}/{sphere}/{dataset}'
    # out_dir = rf'/Users/yuhan/work/nanospheres/data/gas_data_processed/{sphere}/{type}/{dataset}'
    out_dir = rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_data_processed/{sphere}/{type}/{dataset}'
    
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    if fixed_gamma_damping is not None and fixed_c_imp is not None:
        normalized_template, ds_amp2kev = load_cal_from_npz(sphere)
        print(f'  [{dataset}] using npz calibration: amp2kev={ds_amp2kev:.2f}')
        print(f'  [{dataset}] fixed_gamma_damping = {fixed_gamma_damping:.4g} rad/s')
        print(f'  [{dataset}] fixed_c_imp = {fixed_c_imp:.4g}')
    else:
        normalized_template, ds_c_imp_scaling, ds_amp2kev = load_dataset_cal(dataset, type)
        if fixed_gamma_damping is not None:
            print(f'  [{dataset}] fixed_gamma_damping = {fixed_gamma_damping:.4g} rad/s (overrides per-file Voigt fit)')
        if fixed_c_imp is not None:
            print(f'  [{dataset}] fixed_c_imp = {fixed_c_imp:.4g} (overrides c_imp_voigt × c_imp_scaling)')
    ds_sigma_amp = 60 / ds_amp2kev   # chi-square noise in raw amplitude units (~60 keV/c)

    for i in range(nfile):
        outfile_name = f'{data_prefix}{i+idx_start}_processed.hdf5'

        file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
        with h5py.File(file, "r") as f:
            dtt          = f['data'].attrs['delta_t']
            fs           = int(np.ceil(1 / dtt))   # Sampling rate at Hz
            pressure_mbar = f['data'].attrs['pressure_mbar']
            timestamp    = f['data'].attrs['timestamp']
            zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V
            gg = f['data']['channel_g'][:] * f['data']['channel_g'].attrs['adc2mv'] / 1e3  # Signal in V

        # Identify the position of applied impulses
        pulse_indices = utils.get_pulse_idx(gg, trigger_val, positive_pulse)

        zz_notched = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
        zz_bp = utils.bandpass_filtered(zz_notched, fs, bandpass_lb, bandpass_ub, order=lowpass_order)

        if fixed_gamma_damping is not None and fixed_c_imp is not None:
            gamma_damping = fixed_gamma_damping
            c_imp = fixed_c_imp
            p_fit = sv_imp = ffz = sv_z = None
        else:
            # Determine the imprecision noise level from the voigt fit
            p_fit, sv_imp, ffz, sv_z = utils.get_sv_imp(
                            fs, zz, fit_band=(44000, 60000), noise_band=(110000, 120000),
                            nperseg=2**19, p0=[5e-3, 48381*2*np.pi, 100*2*np.pi, 1*2*np.pi])
            A, omega0, sigma, gamma_voigt = p_fit
            gamma_damping = gamma_voigt * 2

            # Scale the imprecision noise in the noise model as doen for calibration data
            c_imp = (np.pi / (omega0**2 * gamma_damping)) * sv_imp / (A)
            c_imp *= ds_c_imp_scaling

            if fixed_gamma_damping is not None:
                gamma_damping = fixed_gamma_damping
            if fixed_c_imp is not None:
                c_imp = fixed_c_imp

        zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / analysis_window_length_gas), analysis_window_length_gas))
        zz_shaped = np.reshape(zz, (int(zz.size / analysis_window_length_gas), analysis_window_length_gas))

        # Minus 3 because trowing away 2/1 amplitudes at the beginning/end of the analysis window
        amp_all         = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length_gas/search_window_length)-3), dtype=np.float64)
        idx_in_window   = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length_gas/search_window_length)-3), dtype=np.int32)
        good_detection  = np.full(shape=zz_bp_shaped.shape[0], fill_value=True)
        chisquare       = np.empty_like(amp_all)
        noise_level_amp = np.empty(shape=zz_bp_shaped.shape[0])
        f_res           = np.empty(shape=zz_bp_shaped.shape[0])
        drive_area      = np.empty(shape=zz_bp_shaped.shape[0])

        lb, ub = 2 * search_window_length, -1 * search_window_length
        for j, _zz_bp in enumerate(zz_bp_shaped):
            amp, amp_lp = utils.recon_force(dtt, _zz_bp, bandpass_ub, lowpass_order, gamma_damping, c_imp)

            # Throw away the beginning and the end of the reconstructed amplitudes
            # to avoid windowing effects
            amp_search = np.abs(amp_lp[lb:ub])
            amp_reshaped = np.reshape(amp_search, (int(amp_search.size/search_window_length), search_window_length))

            amp_searched_idx = np.argmax(amp_reshaped, axis=1)
            amp_searched_idx_in_window = get_idx_in_window(amp_searched_idx, search_window_length, lb)
            amp_all[j] = amp_lp[amp_searched_idx_in_window]
            idx_in_window[j] = amp_searched_idx_in_window

            # Calculate chi2 for each amplitude
            chisquare[j] = calc_chisquares(amp_lp, amp_searched_idx_in_window, normalized_template, sigma_amp=ds_sigma_amp)

            # Noise level in amplitude in the time window
            noise_level_amp[j] = np.std(amp_lp[lb:ub])

            # Identify period of poor detection quality
            if homodyne_lose_lock(zz_shaped[j], zz_bp_shaped[j]):
                good_detection[j] = False

            # Caculate the power of the driven tone
            f_res[j], drive_area[j] = get_driven_power(dtt, zz_shaped[j], notch_freq)

        with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
            print(f'Writing file {os.path.join(out_dir, outfile_name)}')

            g = fout.create_group('data_processed')
            g.attrs['pressure_mbar'] = pressure_mbar
            g.attrs['timestamp'] = timestamp
            g.attrs['amp2kev'] = ds_amp2kev

            if fixed_c_imp is not None:
                g.attrs['fixed_c_imp'] = fixed_c_imp
            if fixed_gamma_damping is not None:
                g.attrs['fixed_gamma_damping'] = fixed_gamma_damping

            # Calculated PSD and Voigt fit parameters
            if p_fit is not None:
                g.create_dataset('ffz', data=ffz, dtype=np.float64)
                g.create_dataset(f'sv_imp', data=np.asarray(sv_imp), dtype=np.float64)
                g.create_dataset(f'voigt_params', data=np.asarray(p_fit), dtype=np.float64)
                g.create_dataset(f'sv_z', data=np.asarray(sv_z), dtype=np.float64)

            g.create_dataset('cal_pulse_indices', data=pulse_indices, dtype=np.int32)
            g.create_dataset('amplitude', data=amp_all, dtype=np.float64)
            g.create_dataset('idx_in_window', data=idx_in_window, dtype=np.int32)

            g.create_dataset('good_detection', data=good_detection, dtype=np.bool_)
            g.create_dataset('noise_level_amp', data=noise_level_amp, dtype=np.float64)
            g.create_dataset('f_res', data=f_res, dtype=np.float64)
            g.create_dataset('driven_power', data=drive_area, dtype=np.float64)
            g.create_dataset('chisquare', data=chisquare, dtype=np.float64)

def scan_c_imp_scaling(prep_data, scan_values, out_path, override_scaling=None):
    """Scan c_imp_scaling values using pre-processed pilot files.

    For each scaling, runs recon_pulse on every triggered pulse in prep_data,
    fits a Gaussian to the amplitude distribution, and selects the scaling
    with the best relative resolution (minimum sigma/mean).

    Parameters
    ----------
    prep_data : list of (dtt, zz_bp, dd, c_imp_voigt, gamma_damping)
        Pre-processed data from pilot files (Voigt fit already done).
    scan_values : list of float
        c_imp_scaling factors to try.
    out_path : str
        Path to save the diagnostic plot.
    override_scaling : float or None
        If provided, marks the corresponding column in the plot as the actual
        value that will be used, distinct from the scan-selected best.

    Returns
    -------
    best_scaling : float
    """
    # pulse sits at analysis_window_length inside f_lp (from get_analysis_window)
    _search_lo = analysis_window_length + 20
    _search_hi = _search_lo + search_window_length

    results = []   # (scaling, amps_raw, waveforms, gp_or_None, a2k_or_None)

    for scaling in scan_values:
        amps, waveforms = [], []
        for (dtt, zz_bp, dd, c_imp_voigt, gamma_damping) in prep_data:
            c_imp = c_imp_voigt * scaling
            for pulse_idx in utils.get_pulse_idx(dd, trigger_val, positive_pulse):
                _, _, f_lp, amp = utils.recon_pulse(
                    pulse_idx, dtt, zz_bp, dd,
                    c_imp=c_imp, gamma_damping=gamma_damping,
                    analysis_window_length=analysis_window_length,
                    prepulse_window_length=fit_window_length,
                    search_window_length=search_window_length,
                    pulse_length=20,
                    lowpass_freq=bandpass_ub,
                    lowpass_order=lowpass_order)
                if np.isnan(amp):
                    continue
                amps.append(amp)

                # Extract waveform centred on the pulse peak
                peak_i   = _search_lo + np.argmax(np.abs(f_lp[_search_lo:_search_hi]))
                polarity = 1 if f_lp[peak_i] > 0 else -1
                lo, hi   = peak_i - waveform_half_len, peak_i + waveform_half_len
                if lo >= 0 and hi <= f_lp.size:
                    waveforms.append(polarity * f_lp[lo:hi])

        amps = np.asarray(amps)
        waveforms = np.asarray(waveforms) if waveforms else np.empty((0, 2 * waveform_half_len))

        gp = None
        if len(amps) > 5:
            bins = np.linspace(0, np.max(amps) * 1.3, 60)
            bc   = 0.5 * (bins[:-1] + bins[1:])
            hh, _ = np.histogram(amps, bins=bins)
            try:
                gp, _ = curve_fit(utils.gauss, bc, hh,
                                   p0=[hh.max(), np.mean(amps), np.std(amps)],
                                   bounds=([0, 0, 0], np.inf), maxfev=10000)
            except Exception as e:
                print(f'  c_imp × {scaling:.3g}: Gaussian fit failed ({e})')

        # Derive amp2kev from Gaussian mean: expected_kev / mu
        a2k = expected_kev / gp[1] if (gp is not None and gp[1] > 0) else None
        results.append((scaling, amps, waveforms, gp, a2k))

    # Select best: minimum relative sigma (sigma / mean)
    best_idx = None
    best_rel_sigma = np.inf
    for i, (_, amps, waveforms, gp, a2k) in enumerate(results):
        if gp is not None and gp[1] > 0:
            rel_sigma = gp[2] / gp[1]
            if rel_sigma < best_rel_sigma:
                best_rel_sigma = rel_sigma
                best_idx = i

    if best_idx is None:
        print('  Warning: Gaussian fit failed for all scalings, keeping default c_imp_scaling')
        best_scaling    = c_imp_scaling
        best_sigma_kev  = None
    else:
        best_scaling = scan_values[best_idx]
        _, _, _, gp_best, a2k_best = results[best_idx]
        best_sigma_kev = gp_best[2] * a2k_best if (gp_best is not None and a2k_best is not None) else None
        print(f'  Best c_imp_scaling = {best_scaling:.4g}  (rel. sigma = {best_rel_sigma:.3f}'
              + (f', σ = {best_sigma_kev:.0f} keV' if best_sigma_kev is not None else '') + ')')

    # ── Plot: 2 rows — waveforms (top), histograms (bottom) ───────────────
    dtt_ref = prep_data[0][0]
    tt_us   = (np.arange(2 * waveform_half_len) - waveform_half_len) * dtt_ref * 1e6

    # Find which column (if any) corresponds to the override value
    override_idx = None
    if override_scaling is not None:
        for i, v in enumerate(scan_values):
            if np.isclose(v, override_scaling):
                override_idx = i
                break

    n_cols = len(scan_values)
    fig, axes = plt.subplots(2, n_cols, figsize=(4 * n_cols, 7), sharey='row')

    for col, (scaling, amps, waveforms, gp, a2k) in enumerate(results):
        ax_w = axes[0, col]
        ax_h = axes[1, col]
        is_override   = (col == override_idx)
        is_scan_best  = (col == best_idx)

        if is_override:
            color = 'darkorange'
        elif is_scan_best:
            color = 'seagreen'
        else:
            color = 'steelblue'

        # Use amp2kev derived from this scaling's Gaussian fit; fall back to config value
        a2k_plot = a2k if a2k is not None else amp2kev_from_cal

        # ── Waveforms ──
        wavs_kev = waveforms * a2k_plot
        for w in wavs_kev:
            ax_w.plot(tt_us, w, color='lightgray', lw=0.6, alpha=0.5)
        if len(wavs_kev):
            ax_w.plot(tt_us, wavs_kev.mean(axis=0), color=color, lw=2)
        ax_w.axhline(0, color='gray', ls='--', lw=0.6)
        ax_w.axvline(0, color='k',    ls=':',  lw=0.8)
        ax_w.set_xlim(-100, 100)
        ax_w.set_xlabel(r'Time ($\mu$s)')
        ax_w.set_ylabel('Recon. amp. (keV/c)')

        title = fr'$c_{{\rm imp}}$ × {scaling:.3g}' + f'\n(n={len(amps)})'
        if is_override:
            title = '★ OVERRIDE\n' + title
        elif is_scan_best:
            title = ('(SCAN BEST)\n' if override_idx is not None else '★ SELECTED\n') + title
        ax_w.set_title(title, fontsize=10)
        if is_override or is_scan_best:
            for spine in ax_w.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        # ── Histogram ──
        amps_kev = amps * a2k_plot
        bins_kev = np.arange(0, max(2500, amps_kev.max() * 1.3 if len(amps_kev) else 2500), 50)
        hh, _    = np.histogram(amps_kev, bins=bins_kev)
        ax_h.stairs(hh, edges=bins_kev, color=color, linewidth=1.5)

        if gp is not None:
            sigma_kev = gp[2] * a2k_plot
            xx = np.linspace(0, bins_kev[-1], 500)
            raw_bin_width = (np.max(amps) * 1.5) / 59 if len(amps) else 1
            kev_bin_width = 50 / a2k_plot
            ax_h.plot(xx, utils.gauss(xx / a2k_plot, *gp) * (kev_bin_width / raw_bin_width),
                      '--', color=color, lw=1.8,
                      label=fr'$\mu$={expected_kev} keV/c, $\sigma$={sigma_kev:.0f} keV/c')
            ax_h.legend(fontsize=8)

        if is_override or is_scan_best:
            for spine in ax_h.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.5)

        ax_h.axvline(expected_kev, color='k', ls=':', lw=1.2)
        ax_h.set_xlim(0, bins_kev[-1])
        ax_h.set_xlabel('Recon. amp. (keV/c)')
        ax_h.set_ylabel('Counts / (50 keV/c)')

    override_note = ''
    if override_scaling is not None:
        if override_idx is not None:
            override_note = f'  |  ★ OVERRIDE applied: {override_scaling:.4g}'
        else:
            override_note = f'  |  OVERRIDE {override_scaling:.4g} (not in scan values)'
    fig.suptitle(
        f'c_imp_scaling scan  ({len(prep_data)} pilot files){override_note}',
        fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Scan plot saved → {out_path}')

    return best_scaling, best_sigma_kev


def _load_and_preprocess_file(data_dir, data_prefix, i, idx_start):
    """Load one raw HDF5, apply notch+bandpass, run Voigt fit.

    Returns (dtt, zz, zz_bp, gg, c_imp_voigt, gamma_damping, p_fit, sv_imp, ffz, sv_z).
    c_imp_voigt is the unscaled imprecision noise parameter.
    """
    file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
    f = h5py.File(file, 'r')

    dtt = f['data'].attrs['delta_t']
    fs  = int(np.ceil(1 / dtt))
    zz  = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3
    gg  = f['data']['channel_g'][:] * f['data']['channel_g'].attrs['adc2mv'] / 1e3
    f.close()

    zz_notched = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
    zz_bp = utils.bandpass_filtered(zz_notched, fs, bandpass_lb, bandpass_ub, order=lowpass_order)

    if fixed_gamma_damping is not None and fixed_c_imp is not None:
        return dtt, zz, zz_bp, gg, fixed_c_imp, fixed_gamma_damping, None, None, None, None

    p_fit, sv_imp, ffz, sv_z = utils.get_sv_imp(
        fs, zz, fit_band=(44000, 60000), noise_band=(110000, 120000),
        nperseg=2**19, p0=[5e-3, 48381*2*np.pi, 100*2*np.pi, 1*2*np.pi])
    A, omega0, sigma, gamma_voigt = p_fit
    gamma_damping = gamma_voigt * 2
    c_imp_voigt = (np.pi / (omega0**2 * gamma_damping)) * sv_imp / A

    return dtt, zz, zz_bp, gg, c_imp_voigt, gamma_damping, p_fit, sv_imp, ffz, sv_z


def process_dataset_cal_mode(sphere, dataset, type, data_prefix, nfile, idx_start):
    """Process a dataset in calibration mode.

    Instead of the sliding-window search, reconstruct each triggered pulse
    individually using utils.recon_pulse() (same as process_impulse_calibration.py).
    All files in the dataset are combined into a single output HDF5:
      {out_dir}/{dataset}_cal_processed.hdf5
    with per-pulse 1-D arrays.

    If SCAN_C_IMP is True, first processes scan_n_files pilot files to find
    the c_imp_scaling with the best amplitude resolution, saves a scan plot,
    then uses that scaling for all files.
    """
    data_dir = rf'/Volumes/LaCie/gas_collisions/{type}/{sphere}/{dataset}'
    out_dir  = rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_data_processed/{sphere}/{type}/{dataset}'

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # ── Optional c_imp_scaling scan on pilot files ────────────────────────
    chosen_scaling = c_imp_scaling   # default from config
    if SCAN_C_IMP and fixed_c_imp is None:
        n_max = min(scan_max_load, nfile)

        # Load files incrementally, stopping as soon as we have scan_n_files good ones.
        # Strategy: load the first scan_n_files to establish an initial median, then
        # load one more at a time only if outliers are preventing us from reaching the target.
        all_prep        = []
        all_c_imp_voigt = []

        def _load_one(idx):
            dtt, zz, zz_bp, gg, c_imp_voigt, gamma_damping, *_ = \
                _load_and_preprocess_file(data_dir, data_prefix, idx, idx_start)
            return (dtt, zz_bp, gg, c_imp_voigt, gamma_damping), c_imp_voigt

        # Initial batch
        n_initial = min(scan_n_files, nfile)
        for i in range(n_initial):
            prep, cv = _load_one(i)
            all_prep.append(prep)
            all_c_imp_voigt.append(cv)

        # Extend one file at a time until we have enough good files or hit n_max
        for i in range(n_initial, n_max):
            c_imp_arr = np.array(all_c_imp_voigt)
            median_c  = np.median(c_imp_arr)
            n_good    = int(np.sum(c_imp_arr <= outlier_c_imp_factor * median_c))
            if n_good >= scan_n_files:
                break
            prep, cv = _load_one(i)
            all_prep.append(prep)
            all_c_imp_voigt.append(cv)

        # Final outlier rejection using the settled median
        c_imp_arr  = np.array(all_c_imp_voigt)
        median_c   = np.median(c_imp_arr)
        good_mask  = c_imp_arr <= outlier_c_imp_factor * median_c
        n_rejected = int(np.sum(~good_mask))
        if n_rejected:
            bad_indices = [i for i, ok in enumerate(good_mask) if not ok]
            print(f'  Rejected {n_rejected} pilot file(s) as Voigt-fit outliers '
                  f'(c_imp_voigt > {outlier_c_imp_factor}× median={median_c:.3e}): '
                  f'file indices {bad_indices}')
        good_prep = [p for p, ok in zip(all_prep, good_mask) if ok]
        if not good_prep:
            print('  Warning: all pilot files rejected — using all unfiltered.')
            good_prep = all_prep
        prep_data = good_prep[:scan_n_files]
        print(f'  Loaded {len(all_prep)} file(s), using {len(prep_data)} good pilot file(s) for scan.')

        scan_plot_path = os.path.join(out_dir, f'{dataset}_c_imp_scan.png')
        chosen_scaling, best_sigma_kev = scan_c_imp_scaling(
            prep_data, c_imp_scan_values, scan_plot_path,
            override_scaling=c_imp_scaling_overrides.get((type, dataset)))

        # Fallback: if best sigma is still too wide, the scan couldn't find a clean peak
        if best_sigma_kev is not None and best_sigma_kev > sigma_kev_threshold:
            print(f'  Warning: best scan σ = {best_sigma_kev:.0f} keV exceeds threshold '
                  f'{sigma_kev_threshold} keV — falling back to default c_imp_scaling = {c_imp_scaling:.4g}')
            chosen_scaling = c_imp_scaling

    override = c_imp_scaling_overrides.get((type, dataset))
    if fixed_c_imp is not None:
        print(f'  fixed_c_imp = {fixed_c_imp:.4g} — c_imp_scaling scan/override skipped')
    elif override is not None:
        print(f'  c_imp_scaling override: {override:.4g} (scan/default had {chosen_scaling:.4g})')
        chosen_scaling = override
    else:
        print(f'  Using c_imp_scaling = {chosen_scaling:.4g}')

    if fixed_gamma_damping is not None:
        print(f'  fixed_gamma_damping = {fixed_gamma_damping:.4g} rad/s (overrides per-file Voigt fit)')

    # ── Main loop: process all files ──────────────────────────────────────
    amps, idxs_in_win, f_res_list = [], [], []
    drive_area_list, noise_level_list = [], []
    file_idx_list, pulse_abs_idx_list = [], []
    waveforms = []

    _search_lo = analysis_window_length + 20
    _search_hi = _search_lo + search_window_length

    sv_imps, voigt_params_list, sv_zs = [], [], []
    ffz_saved = None

    for i in range(nfile):
        dtt, zz, zz_bp, gg, c_imp_voigt, gamma_damping, p_fit, sv_imp, ffz, sv_z = \
            _load_and_preprocess_file(data_dir, data_prefix, i, idx_start)

        if fixed_c_imp is not None:
            c_imp = fixed_c_imp
        else:
            c_imp = c_imp_voigt * chosen_scaling
        if fixed_gamma_damping is not None:
            gamma_damping = fixed_gamma_damping

        if sv_imp is not None:
            sv_imps.append(sv_imp)
            voigt_params_list.append(p_fit)
            sv_zs.append(sv_z)
            if ffz_saved is None:
                ffz_saved = ffz

        pulse_indices = utils.get_pulse_idx(gg, trigger_val, positive_pulse)

        for pulse_idx in pulse_indices:
            window, _, f_lp, amp = utils.recon_pulse(
                pulse_idx, dtt, zz_bp, dd=gg, c_imp=c_imp, gamma_damping=gamma_damping,
                analysis_window_length=analysis_window_length,
                prepulse_window_length=fit_window_length,
                search_window_length=search_window_length,
                pulse_length=20,
                lowpass_freq=bandpass_ub,
                lowpass_order=lowpass_order)
            if window is None:
                continue

            # Resonant frequency and driven power from prepulse window
            prepulse_win = utils.get_prepulse_window(gg, pulse_idx, fit_window_length)
            f_res_val, drive_area_val = get_driven_power(dtt, zz[prepulse_win], notch_freq)

            # Pulse index within the reconstructed window
            _, pulse_idx_in_win = utils.get_analysis_window(gg, pulse_idx, analysis_window_length)

            # Noise level: std of f_lp excluding edges
            _lb, _ub = 2 * search_window_length, -1 * search_window_length
            noise_level = np.std(f_lp[_lb:_ub])

            # Waveform centred on the pulse peak
            peak_i   = _search_lo + np.argmax(np.abs(f_lp[_search_lo:_search_hi]))
            polarity = 1 if f_lp[peak_i] > 0 else -1
            lo, hi   = peak_i - waveform_half_len, peak_i + waveform_half_len
            if lo < 0 or hi > f_lp.size:
                print(f'  Skipping pulse near edge (file {i}, pulse_idx {pulse_idx})')
                continue
            waveforms.append(polarity * f_lp[lo:hi])

            amps.append(amp)
            idxs_in_win.append(pulse_idx_in_win)
            f_res_list.append(f_res_val)
            drive_area_list.append(drive_area_val)
            noise_level_list.append(noise_level)
            file_idx_list.append(i)
            pulse_abs_idx_list.append(pulse_idx)

        print(f'  file {i+idx_start}: {len(pulse_indices)} pulses found')

    outfile = os.path.join(out_dir, f'{dataset}_cal_processed.hdf5')
    with h5py.File(outfile, 'w') as fout:
        print(f'Writing {outfile}')
        g = fout.create_group('data_processed')

        if fixed_c_imp is not None:
            g.attrs['fixed_c_imp'] = fixed_c_imp
        else:
            g.attrs['c_imp_scaling'] = chosen_scaling
        if fixed_gamma_damping is not None:
            g.attrs['fixed_gamma_damping'] = fixed_gamma_damping

        if sv_imps:
            if ffz_saved is not None:
                g.create_dataset('ffz',          data=ffz_saved,                     dtype=np.float64)
            g.create_dataset('sv_imp',           data=np.asarray(sv_imps),           dtype=np.float64)
            g.create_dataset('voigt_params',     data=np.asarray(voigt_params_list), dtype=np.float64)
            g.create_dataset('sv_z',             data=np.asarray(sv_zs),             dtype=np.float64)

        g.create_dataset('amplitude',        data=np.asarray(amps),              dtype=np.float64)
        g.create_dataset('waveform',         data=np.asarray(waveforms),         dtype=np.float64)
        g.create_dataset('idx_in_window',    data=np.asarray(idxs_in_win),       dtype=np.int32)
        g.create_dataset('f_res',            data=np.asarray(f_res_list),        dtype=np.float64)
        g.create_dataset('driven_power',     data=np.asarray(drive_area_list),   dtype=np.float64)
        g.create_dataset('noise_level_amp',  data=np.asarray(noise_level_list),  dtype=np.float64)
        g.create_dataset('file_index',       data=np.asarray(file_idx_list),     dtype=np.int32)
        g.create_dataset('pulse_abs_index',  data=np.asarray(pulse_abs_idx_list),dtype=np.int32)


if __name__ == '__main__':
    import sys
    # Usage:
    #   python process_gas_data.py [--cal | --no-cal] [gas_type ...]
    #
    # --cal / --no-cal  override CAL_MODE_DEFAULT from config
    # gas_type ...      restrict to one or more gas types (default: all)
    #
    # Examples:
    #   python process_gas_data.py                     # use CAL_MODE_DEFAULT from config
    #   python process_gas_data.py --cal xenon         # cal mode, xenon only
    #   python process_gas_data.py --no-cal xenon sf6  # non-cal mode, xenon + sf6

    args = sys.argv[1:]
    if '--cal' in args:
        run_cal_mode = True
        args.remove('--cal')
    elif '--no-cal' in args:
        run_cal_mode = False
        args.remove('--no-cal')
    else:
        run_cal_mode = CAL_MODE_DEFAULT

    gas_filter = set(args) if args else set(datasets_config)

    print(f'CAL_MODE = {run_cal_mode}')
    for gas_type, entries in datasets_config.items():
        if gas_type not in gas_filter:
            continue
        for dataset, data_type, file_prefix, nfiles in entries:
            print(f'\n[{gas_type}] {dataset}')
            if run_cal_mode:
                process_dataset_cal_mode(sphere, dataset, data_type, file_prefix, nfiles, idx_start=0)
            else:
                process_dataset(sphere, dataset, data_type, file_prefix, nfiles, idx_start=0)