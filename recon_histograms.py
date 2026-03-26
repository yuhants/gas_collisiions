import numpy as np
import os
import h5py

# ============================================================
# CONFIG — edit these before running
# ============================================================
sphere = 'sphere_20260215'
# amp2kev_from_cal = 8363.560351624732   # sphere_20260215
amp2kev_from_cal = 10497.219118653622    # sphere_20260215; after introducing imprecision

# Path to processed gas data
# processed_data_dir = r'/Users/yuhan/work/nanospheres/data/gas_data_processed' # directory for old noise model
processed_data_dir = rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_data_processed'

# Output directory and filename
outdir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_recon'
outfile_name = f'{sphere}_gas_recon_all.h5py'

# Histogram bins (keV/c)
hist_bins = np.arange(0, 2000, 25)

# Quality cut parameters
noise_threshold_kev = 100
chi2_threshold = 1000
normalized_drive_power_threshold = 4.5e-9

# Analysis window structure (must match process_gas_data.py)
analysis_window_length = 2**19
search_window_length   = 2**8
lb                     = 2 * search_window_length

# Double-count removal parameters
doublecount_amp_thr_kev          = 0    # min amplitude for same-peak pairs (keV)
doublecount_opp_sign_amp_thr_kev = 180  # min amplitude for opposite-sign peak+trough pairs (keV)
doublecount_idx_thr              = 25   # max index separation for same-peak pairs
doublecount_opp_sign_idx_thr     = search_window_length // 3  # max index separation for opposite-sign pairs

# Calibration-pulse identification parameters
cal_pulse_amp_thr_kev = 500   # applied impulses are ~1100 keV/c; flag above this threshold
# Peak is expected in [pulse_index + offset, pulse_index + offset + search_window_length]
# The +20 offset comes from get_search_window() in analysis_utils.py (pulse_length=20)
cal_pulse_offset = 20

# Drive tone frequency (Hz)
drive_freq = 137000
# Reference frequency (Hz) — used for normalizing driven power
ref_freq = 48500

# Datasets grouped by gas type.
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
        ('20260219_p6e_4e-7mbar_1', 'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_2e-7mbar',   'xenon_data', '20260219_dfg_', 25),
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
        ('20260219_p6e_3e-7mbar_1', 'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-8mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-8mbar',   'sf6_data', '20260219_dfg_', 25),
    ],
}
# ============================================================


def read_recon(file):
    with h5py.File(file, 'r') as f:
        g              = f['data_processed']
        pressure       = g.attrs['pressure_mbar']
        file_amp2kev   = float(g.attrs.get('amp2kev', amp2kev_from_cal))
        amps           = g['amplitude'][:]
        idx_in_window  = g['idx_in_window'][:]
        good_detection = g['good_detection'][:]
        noise_level_amp = g['noise_level_amp'][:]
        chi2           = g['chisquare'][:]
        f_res          = g['f_res'][:]
        driven_power   = g['driven_power'][:]
        pulse_indices  = g['cal_pulse_indices'][:]

    return amps, good_detection, noise_level_amp, chi2, f_res, driven_power, pressure, idx_in_window, pulse_indices, file_amp2kev


def read_recon_all(dataset, data_type, file_prefix, nfiles):
    amps_all, good_detection_all, noise_level_all = [], [], []
    chi2_all, driven_power_all, f_res_all, pressure_all, idx_in_window_all, pulse_indices_all, amp2kev_all = [], [], [], [], [], [], []

    for i in range(nfiles):
        file = os.path.join(
            processed_data_dir, sphere, data_type, dataset,
            f'{file_prefix}{i}_processed.hdf5'
        )
        amps, good_detection, noise_level_amp, chi2, f_res, driven_power, pressure, idx_in_window, pulse_indices, file_amp2kev = read_recon(file)
        amps_all.append(amps)
        good_detection_all.append(good_detection)
        noise_level_all.append(noise_level_amp)
        chi2_all.append(chi2)
        driven_power_all.append(driven_power)
        f_res_all.append(f_res)
        pressure_all.append(pressure)
        idx_in_window_all.append(idx_in_window)
        pulse_indices_all.append(pulse_indices)
        amp2kev_all.append(file_amp2kev)
        # print(pressure)

    # pressure_all stays at index 6, amp2kev_all at index 9
    return amps_all, good_detection_all, noise_level_all, chi2_all, driven_power_all, f_res_all, pressure_all, idx_in_window_all, pulse_indices_all, amp2kev_all


def read_recon_mc_file(filepath):
    """
    Read a single MC output HDF5 file (from gen_signal_model_mc.py) and return
    a recon_output tuple compatible with get_summed_histogram.

    Parameters
    ----------
    filepath : str
        Path to an MC HDF5 file produced by gen_signal_model_mc.py.

    Returns
    -------
    recon_output : tuple
        Same 10-element format as read_recon_all: each element is a length-1
        list wrapping the corresponding array.  Compatible with
        get_summed_histogram without modification.
    mc_attrs : dict
        MC-specific metadata (mc_gas, mc_T_sensor, mc_alpha, mc_n_windows, …).
    mc_injected : dict or None
        If the file contains 'mc_injected' group (truth info), a dict with:
          'q_true_kev'  : (N,) float64 — true |qz| in keV/c
          'positions'   : (N,) int32   — peak sample position in analysis window
          'signs'       : (N,) int8    — ±1 injection sign
          'window_idx'  : (N,) int32   — analysis window index
        Sub-window index: (positions - lb) // search_window_length.
        None if the group is absent (older MC files).
    """
    row = read_recon(filepath)   # (amps, good_det, noise_lv, chi2, f_res, drv_pwr, p, idx, cal_idx, a2k)
    recon_output = tuple([x] for x in row)

    with h5py.File(filepath, 'r') as f:
        mc_attrs = {k: v for k, v in f['data_processed'].attrs.items()
                    if k.startswith('mc_')}

        if 'mc_injected' in f:
            ig = f['mc_injected']
            mc_injected = {k: ig[k][:] for k in ig}
        else:
            mc_injected = None

    return recon_output, mc_attrs, mc_injected


def throw_away_doublecounts(amplitude, good_det_noise, idx_in_window,
                            file_amp2kev=None,
                            amp_thr_kev=doublecount_amp_thr_kev,
                            opp_sign_amp_thr_kev=doublecount_opp_sign_amp_thr_kev,
                            double_count_idx_thr=doublecount_idx_thr,
                            opp_sign_idx_thr=doublecount_opp_sign_idx_thr):
    """
    Identify and null out double-counted pulses.

    Two cases are handled, each with its own amplitude and index thresholds:

    1. Same-peak straddling (idx separation < double_count_idx_thr,
       both amplitudes > amp_thr_kev):
       The pulse peak sits near a sub-window boundary so both adjacent windows
       find the same peak.  Indices are nearly identical (< ~25 samples).

    2. Peak + ring-down trough (opposite sign, idx separation < opp_sign_idx_thr,
       both amplitudes > opp_sign_amp_thr_kev):
       One sub-window captures the positive peak of a pulse; a nearby sub-window
       captures the negative ring-down trough of the same pulse.  The trough
       typically appears ~T/2 after the peak (≈50 samples at 50 kHz / 5 MHz).

    In both cases the smaller-|amplitude| detection is set to NaN; the larger
    is kept.

    Parameters
    ----------
    amplitude : (n_windows, n_searches) float array
        Reconstructed pulse amplitudes (in amplitude units, not keV).
    good_det_noise : (n_windows,) bool array
        True for windows passing quality and noise cuts.
    idx_in_window : (n_windows, n_searches) int array
        Absolute sample index of each peak within its analysis window.
    amp_thr_kev : float
        Min amplitude (keV/c) for same-peak straddling pairs.
    opp_sign_amp_thr_kev : float
        Min amplitude (keV/c) for opposite-sign peak+trough pairs.
    double_count_idx_thr : int
        Max index separation for same-peak straddling pairs.
    opp_sign_idx_thr : int
        Max index separation for opposite-sign peak+trough pairs.

    Returns
    -------
    ret : (n_windows, n_searches) float array
        Copy of amplitude with double-count duplicates set to NaN.
    """
    ret = np.array(amplitude, dtype=np.float64)
    n_searches = amplitude.shape[1]
    a2k = file_amp2kev if file_amp2kev is not None else amp2kev_from_cal

    # Include any pulse that could qualify under either condition
    min_amp_thr = min(amp_thr_kev, opp_sign_amp_thr_kev)
    large_pulses = (
        (np.abs(ret) * a2k > min_amp_thr)
        & np.tile(good_det_noise[:, np.newaxis], (1, n_searches))
    )

    abs_idx_large    = idx_in_window[large_pulses].astype(np.int64)
    idx_sep          = np.abs(np.diff(abs_idx_large, append=analysis_window_length))
    large_pulses_pos = np.argwhere(large_pulses)   # shape (N, 2)
    amplitude_large  = ret[large_pulses]            # 1-D view of the selected values

    # Case 1: any-sign pair with small index separation
    same_peak = idx_sep < double_count_idx_thr

    # Case 2: opposite-sign pair within opp_sign_idx_thr (peak + ring-down trough)
    signs       = np.sign(amplitude_large)
    opp_sign    = np.append(signs[:-1] != signs[1:], False)
    peak_trough = opp_sign & (idx_sep < opp_sign_idx_thr)

    for i in range(len(large_pulses_pos) - 1):
        if np.isnan(amplitude_large[i]):
            continue

        # Skip pairs that span different analysis windows (cross-window false positive)
        if large_pulses_pos[i][0] != large_pulses_pos[i + 1][0]:
            continue

        # Check each condition with its own amplitude threshold
        min_pair_amp_kev = min(abs(amplitude_large[i]), abs(amplitude_large[i + 1])) * a2k
        sp = same_peak[i]   and (min_pair_amp_kev > amp_thr_kev)
        pt = peak_trough[i] and (min_pair_amp_kev > opp_sign_amp_thr_kev)
        if not sp and not pt:
            continue

        # Null out the smaller-amplitude detection; keep amplitude_large in sync
        # so the NaN guard above works correctly for subsequent iterations.
        if np.abs(amplitude_large[i]) < np.abs(amplitude_large[i + 1]):
            ret[large_pulses_pos[i][0], large_pulses_pos[i][1]] = np.nan
            amplitude_large[i] = np.nan
        else:
            ret[large_pulses_pos[i + 1][0], large_pulses_pos[i + 1][1]] = np.nan
            amplitude_large[i + 1] = np.nan

    return ret


def flag_cal_pulses(idx_in_window, pulse_indices, amplitude,
                    file_amp2kev=None,
                    amp_thr_kev=cal_pulse_amp_thr_kev,
                    offset=cal_pulse_offset):
    """
    Return a boolean mask (n_windows, n_searches) that is True where a
    detected amplitude is timing-coincident with an applied calibration pulse
    AND has amplitude above the calibration pulse threshold.

    The nearest cal pulse trigger is found in either direction. timing_match is
    True when the detected peak falls within ±search_window_length samples of
    pulse_index + offset, i.e. |abs_idx - nearest_pulse - offset| < search_window_length.

    Parameters
    ----------
    idx_in_window : (n_windows, n_searches) int32 array
        Absolute peak positions within each analysis window.
    pulse_indices : 1-D int array
        Sample indices in the full signal where calibration pulses were applied.
    amplitude : (n_windows, n_searches) float array
        Reconstructed pulse amplitudes in native units (not keV).
    amp_thr_kev : float
        Minimum amplitude (keV/c) for a detection to be considered a cal pulse
        (default 700 keV/c; applied impulses are ~1100 keV/c).
    offset : int
        Sample offset from trigger to centre of acceptance window (default 20).

    Returns
    -------
    is_cal : (n_windows, n_searches) bool array
    """
    if pulse_indices.size == 0:
        return np.zeros(idx_in_window.shape, dtype=bool)

    n_windows = idx_in_window.shape[0]
    window_offsets = np.arange(n_windows, dtype=np.int64) * analysis_window_length
    abs_idx = (window_offsets[:, np.newaxis] + idx_in_window.astype(np.int64)).ravel()

    sorted_pulses = np.sort(pulse_indices.astype(np.int64))
    # For each detected peak, find the nearest cal pulse trigger (left or right)
    pos = np.searchsorted(sorted_pulses, abs_idx, side='left')
    left_pos  = np.clip(pos - 1, 0, sorted_pulses.size - 1)
    right_pos = np.clip(pos,     0, sorted_pulses.size - 1)
    left_pulse  = sorted_pulses[left_pos]
    right_pulse = sorted_pulses[right_pos]
    has_left  = pos > 0
    has_right = pos < sorted_pulses.size
    nearest_pulse = np.where(
        has_left & has_right,
        np.where(np.abs(abs_idx - left_pulse) <= np.abs(abs_idx - right_pulse),
                 left_pulse, right_pulse),
        np.where(has_left, left_pulse, right_pulse),
    )
    delta = abs_idx - nearest_pulse
    timing_match = np.abs(delta - offset) < search_window_length

    a2k = file_amp2kev if file_amp2kev is not None else amp2kev_from_cal
    amp_match = np.abs(amplitude.ravel()) * a2k > amp_thr_kev
    return (timing_match & amp_match).reshape(idx_in_window.shape)


def get_summed_histogram(recon_output, bins, remove_doublecounts=True,
                         apply_noise_cut=True, apply_drive_cut=True, apply_chi2_cut=True):
    """
    Build summed amplitude histograms over all files in recon_output.

    Intermediate histograms are accumulated after each cumulative selection
    stage so the effect of every cut can be inspected independently.
    Cal-pulse flagging is done on raw amplitudes (before quality cuts) and
    the nocal/cal split is provided at every stage.

    Parameters
    ----------
    apply_noise_cut : bool
        Apply noise_threshold_kev cut on noise_level_amp (default True).
    apply_drive_cut : bool
        Apply normalized_drive_power_threshold cut (default True).
    apply_chi2_cut : bool
        Null amplitudes where chi2 >= chi2_threshold (default True).

    Returns
    -------
    bc : bin centres (keV/c)
    histograms : dict of str → int64 array
        Keys (cumulative selection stages):
        'nocuts'        — no cuts at all
        'hom_bal'       — homodyne balance (good_detection) only
        'noise'         — + noise threshold
        'drive'         — + drive power threshold
        'chi2'          — + chi2 threshold
        'all'           — + double-count removal
        Append '_nocal' or '_cal' for the cal-pulse split at each stage.
    """
    amps_all, good_detection_all, noise_level_all, chi2_all, driven_power_all, f_res_all = recon_output[:6]
    idx_in_window_all  = recon_output[7] if len(recon_output) > 7 else None
    pulse_indices_all  = recon_output[8] if len(recon_output) > 8 else None
    amp2kev_all        = recon_output[9] if len(recon_output) > 9 else [amp2kev_from_cal] * len(amps_all)

    bc = 0.5 * (bins[:-1] + bins[1:])
    stage_names = ['nocuts', 'hom_bal', 'noise', 'drive', 'chi2', 'all']
    hh = {}
    for s in stage_names:
        hh[s]           = np.zeros_like(bc, dtype=np.int64)
        hh[s + '_nocal'] = np.zeros_like(bc, dtype=np.int64)
        hh[s + '_cal']   = np.zeros_like(bc, dtype=np.int64)

    def _hist(amps_2d, mask_1d):
        vals = amps_2d[mask_1d].ravel()
        vals = vals[~np.isnan(vals)]
        h, _ = np.histogram(np.abs(vals) * a2k, bins)
        return h

    def _accumulate(stage, amps_2d, mask_1d, is_cal):
        h = _hist(amps_2d, mask_1d)
        hh[stage] += h
        if is_cal is not None:
            hh[stage + '_nocal'] += _hist(np.where(is_cal, np.nan, amps_2d), mask_1d)
            hh[stage + '_cal']   += _hist(np.where(is_cal, amps_2d, np.nan), mask_1d)
        else:
            hh[stage + '_nocal'] += h

    for i in range(len(good_detection_all)):
        a2k = amp2kev_all[i]
        amps_raw = np.array(amps_all[i], dtype=np.float64)

        # Flag cal pulses on raw amplitudes (before any cuts)
        if idx_in_window_all is not None and pulse_indices_all is not None:
            is_cal = flag_cal_pulses(idx_in_window_all[i], pulse_indices_all[i], amps_raw, file_amp2kev=a2k)
        else:
            is_cal = None

        # --- Stage: nocuts ---
        mask_all = np.ones(len(good_detection_all[i]), dtype=bool)
        _accumulate('nocuts', amps_raw, mask_all, is_cal)

        # --- Stage: hom_bal (good_detection) ---
        mask_hom = good_detection_all[i]
        _accumulate('hom_bal', amps_raw, mask_hom, is_cal)

        # --- Stage: + noise ---
        if apply_noise_cut:
            noise_ok = noise_level_all[i] * a2k < noise_threshold_kev
        else:
            noise_ok = np.ones(len(good_detection_all[i]), dtype=bool)
        mask_noise = mask_hom & noise_ok
        _accumulate('noise', amps_raw, mask_noise, is_cal)

        # --- Stage: + drive ---
        if apply_drive_cut:
            norm_drive = (
                driven_power_all[i]
                * (f_res_all[i]**2 - drive_freq**2)**2
                / (ref_freq**2 - drive_freq**2)**2
            )
            drive_ok = norm_drive > normalized_drive_power_threshold
        else:
            drive_ok = np.ones(len(good_detection_all[i]), dtype=bool)
        mask_drive = mask_noise & drive_ok
        _accumulate('drive', amps_raw, mask_drive, is_cal)

        # --- Stage: + chi2 ---
        amps_chi2 = np.array(amps_raw)
        if apply_chi2_cut:
            amps_chi2[chi2_all[i] >= chi2_threshold] = np.nan
        _accumulate('chi2', amps_chi2, mask_drive, is_cal)

        # --- Stage: + double-count removal ---
        if remove_doublecounts and idx_in_window_all is not None:
            amps_final = throw_away_doublecounts(amps_chi2, mask_drive, idx_in_window_all[i], file_amp2kev=a2k)
        else:
            amps_final = amps_chi2
        _accumulate('all', amps_final, mask_drive, is_cal)

    return bc, hh


def mean_pressure(recon_output):
    # Reject pressure that are negative, which is likely due to
    # problematic readout of the pressure gauge
    p_all = np.asarray(recon_output[6])
    return float(np.mean(p_all[p_all > 0]))


if __name__ == '__main__':
    import sys
    # Usage:
    #   python recon_histograms.py [--no-noise-cut] [--no-drive-cut] [--no-chi2-cut] [gas_type ...]
    #
    # --no-noise-cut   skip noise_threshold_kev cut
    # --no-drive-cut   skip normalized_drive_power_threshold cut
    # --no-chi2-cut    skip chi2_threshold cut
    # gas_type ...     restrict to one or more gas types (default: all)
    #
    # Examples:
    #   python recon_histograms.py xenon sf6
    #   python recon_histograms.py --no-noise-cut --no-drive-cut xenon

    args = sys.argv[1:]

    apply_noise_cut = '--no-noise-cut' not in args
    apply_drive_cut = '--no-drive-cut' not in args
    apply_chi2_cut  = '--no-chi2-cut'  not in args
    for flag in ('--no-noise-cut', '--no-drive-cut', '--no-chi2-cut'):
        if flag in args:
            args.remove(flag)

    gas_filter = set(args) if args else set(datasets_config)

    unknown = gas_filter - set(datasets_config)
    if unknown:
        print(f'Unknown gas type(s): {", ".join(sorted(unknown))}')
        print(f'Valid types: {", ".join(sorted(datasets_config))}')
        sys.exit(1)

    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile_name)
    print(f'Output: {outpath}')
    print(f'Processing: {", ".join(sorted(gas_filter))}')
    cuts_active = []
    if apply_noise_cut:  cuts_active.append(f'noise<{noise_threshold_kev}keV')
    if apply_drive_cut:  cuts_active.append(f'drive>{normalized_drive_power_threshold:.1e}')
    if apply_chi2_cut:   cuts_active.append(f'chi2<{chi2_threshold}')
    print(f'Cuts: {", ".join(cuts_active) if cuts_active else "none"}')

    # Open in append mode so unprocessed gas types are preserved
    with h5py.File(outpath, 'a') as fout:
        if 'recon_histograms' not in fout:
            g = fout.create_group('recon_histograms')
            bc = 0.5 * (hist_bins[:-1] + hist_bins[1:])
            d = g.create_dataset('bc', data=bc, dtype=np.float64)
            d.attrs['unit'] = 'keV/c'
        else:
            g = fout['recon_histograms']

        for gas_type, entries in datasets_config.items():
            if gas_type not in gas_filter:
                continue

            # Replace the group if it already exists
            if gas_type in g:
                del g[gas_type]
            grp = g.create_group(gas_type)

            for dataset, data_type, file_prefix, nfiles in entries:

                print(f'  {gas_type}/{dataset} ({nfiles} files)...')
                recon = read_recon_all(dataset, data_type, file_prefix, nfiles)
                bc, histograms = get_summed_histogram(
                    recon, hist_bins, remove_doublecounts=True,
                    apply_noise_cut=apply_noise_cut,
                    apply_drive_cut=apply_drive_cut,
                    apply_chi2_cut=apply_chi2_cut,
                )

                unit = f'count/{hist_bins[1]-hist_bins[0]:.0f}keV'
                p = mean_pressure(recon)

                for key, data in histograms.items():
                    # 'all' stage uses bare dataset name; 'all_nocal'/'all_cal'
                    # drop the 'all_' prefix for backward compatibility
                    if key == 'all':
                        name = dataset
                    elif key.startswith('all_'):
                        name = f'{dataset}_{key[4:]}'  # all_nocal → _nocal
                    else:
                        name = f'{dataset}_{key}'
                    d = grp.create_dataset(name, data=data, dtype=np.int64)
                    d.attrs['pressure_mbar'] = p
                    d.attrs['unit'] = unit

    print('Done.')
