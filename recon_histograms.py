import numpy as np
import os
import h5py

# ============================================================
# CONFIG — edit these before running
# ============================================================
sphere = 'sphere_20260215'
amp2kev = 8363.560351624732

# Path to processed gas data
processed_data_dir = r'/Users/yuhan/work/nanospheres/data/gas_data_processed'

# Output directory and filename
outdir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/gas_recon'
outfile_name = f'{sphere}_gas_recon_all.h5py'

# Histogram bins (keV/c)
hist_bins = np.arange(0, 2000, 25)

# Quality cut parameters
noise_threshold_kev = 70
chi2_threshold = 700
normalized_drive_power_threshold = 4.5e-9

# Analysis window structure (must match process_gas_data.py)
analysis_window_length = 2**19
search_window_length   = 2**8
lb                     = 2 * search_window_length

# Double-count removal parameters
doublecount_amp_thr_kev          = 0   # min amplitude for same-peak pairs (keV)
doublecount_opp_sign_amp_thr_kev = 180  # min amplitude for opposite-sign peak+trough pairs (keV)
doublecount_idx_thr              = 25  # max index separation for same-peak pairs
doublecount_opp_sign_idx_thr     = search_window_length // 3  # max index separation for opposite-sign pairs

# Calibration-pulse identification parameters
cal_pulse_amp_thr_kev = 700   # applied impulses are ~1100 keV/c; flag above this threshold
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
        pressure = f['data_processed'].attrs['pressure_mbar']
        amps           = f['data_processed']['amplitude'][:]
        idx_in_window  = f['data_processed']['idx_in_window'][:]
        good_detection = f['data_processed']['good_detection'][:]
        noise_level_amp = f['data_processed']['noise_level_amp'][:]
        chi2           = f['data_processed']['chisquare'][:]
        f_res          = f['data_processed']['f_res'][:]
        driven_power   = f['data_processed']['driven_power'][:]
        pulse_indices  = f['data_processed']['pulse_indices'][:]
    return amps, good_detection, noise_level_amp, chi2, f_res, driven_power, pressure, idx_in_window, pulse_indices


def read_recon_all(dataset, data_type, file_prefix, nfiles):
    amps_all, good_detection_all, noise_level_all = [], [], []
    chi2_all, driven_power_all, f_res_all, pressure_all, idx_in_window_all, pulse_indices_all = [], [], [], [], [], []

    for i in range(nfiles):
        file = os.path.join(
            processed_data_dir, sphere, data_type, dataset,
            f'{file_prefix}{i}_processed.hdf5'
        )
        amps, good_detection, noise_level_amp, chi2, f_res, driven_power, pressure, idx_in_window, pulse_indices = read_recon(file)
        amps_all.append(amps)
        good_detection_all.append(good_detection)
        noise_level_all.append(noise_level_amp)
        chi2_all.append(chi2)
        driven_power_all.append(driven_power)
        f_res_all.append(f_res)
        pressure_all.append(pressure)
        idx_in_window_all.append(idx_in_window)
        pulse_indices_all.append(pulse_indices)

    # pressure_all stays at index 6 so mean_pressure() is unaffected
    return amps_all, good_detection_all, noise_level_all, chi2_all, driven_power_all, f_res_all, pressure_all, idx_in_window_all, pulse_indices_all


def throw_away_doublecounts(amplitude, good_det_noise, idx_in_window,
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

    # Include any pulse that could qualify under either condition
    min_amp_thr = min(amp_thr_kev, opp_sign_amp_thr_kev)
    large_pulses = (
        (np.abs(ret) * amp2kev > min_amp_thr)
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
        min_pair_amp_kev = min(abs(amplitude_large[i]), abs(amplitude_large[i + 1])) * amp2kev
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
                    amp_thr_kev=cal_pulse_amp_thr_kev,
                    offset=cal_pulse_offset):
    """
    Return a boolean mask (n_windows, n_searches) that is True where a
    detected amplitude is timing-coincident with an applied calibration pulse
    AND has amplitude above the calibration pulse threshold.

    The expected peak position is [pulse_index + offset,
    pulse_index + offset + search_window_length), matching the search window
    used by get_search_window() in analysis_utils.py (pulse_length=20).

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
        Sample offset from trigger to start of search window (default 20).

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
    # For each detected peak, find the nearest pulse trigger to the left
    pos = np.searchsorted(sorted_pulses, abs_idx, side='right') - 1

    timing_match = np.zeros(abs_idx.shape, dtype=bool)
    valid = (pos >= 0) & (pos < sorted_pulses.size)
    delta = np.where(valid, abs_idx - sorted_pulses[np.clip(pos, 0, sorted_pulses.size - 1)],
                     np.int64(-1))
    timing_match = (delta >= offset) & (delta < offset + search_window_length)

    amp_match = np.abs(amplitude.ravel()) * amp2kev > amp_thr_kev
    return (timing_match & amp_match).reshape(idx_in_window.shape)


def get_summed_histogram(recon_output, bins, remove_doublecounts=True):
    """
    Build summed amplitude histograms over all files in recon_output.

    Returns
    -------
    bc : bin centres (keV/c)
    hh_all : histogram of all passing amplitudes
    hh_nocal : histogram with calibration-pulse coincident amplitudes removed
    """
    amps_all, good_detection_all, noise_level_all, chi2_all, driven_power_all, f_res_all = recon_output[:6]
    idx_in_window_all  = recon_output[7] if len(recon_output) > 7 else None
    pulse_indices_all  = recon_output[8] if len(recon_output) > 8 else None

    bc = 0.5 * (bins[:-1] + bins[1:])
    hh_all   = np.zeros_like(bc, dtype=np.int64)
    hh_nocal = np.zeros_like(bc, dtype=np.int64)

    # Looping over all files in each dataset and perform data selection
    for i in range(len(good_detection_all)):
        noise_ok = noise_level_all[i] * amp2kev < noise_threshold_kev
        norm_drive = (
            driven_power_all[i]
            * (f_res_all[i]**2 - drive_freq**2)**2
            / (ref_freq**2 - drive_freq**2)**2
        )
        good_window = good_detection_all[i] & noise_ok & (norm_drive > normalized_drive_power_threshold)

        # Apply chi2 cut by nulling bad entries (keeps 2-D shape for subsequent steps)
        amps = np.array(amps_all[i], dtype=np.float64)
        amps[chi2_all[i] >= chi2_threshold] = np.nan

        if remove_doublecounts and idx_in_window_all is not None:
            amps = throw_away_doublecounts(amps, good_window, idx_in_window_all[i])

        good_amps = amps[good_window].ravel()
        passing_amps = good_amps[~np.isnan(good_amps)]
        hh, _ = np.histogram(np.abs(passing_amps) * amp2kev, bins)
        hh_all += hh

        # Cal-pulse mask: flag entries that are timing-coincident AND above amplitude threshold
        if idx_in_window_all is not None and pulse_indices_all is not None:
            is_cal = flag_cal_pulses(idx_in_window_all[i], pulse_indices_all[i], amps)
            amps_nocal = np.where(is_cal, np.nan, amps)
            good_amps_nocal = amps_nocal[good_window].ravel()
            passing_nocal = good_amps_nocal[~np.isnan(good_amps_nocal)]
            hh_nc, _ = np.histogram(np.abs(passing_nocal) * amp2kev, bins)
            hh_nocal += hh_nc
        else:
            hh_nocal += hh

    return bc, hh_all, hh_nocal


def mean_pressure(recon_output):
    return float(np.mean(recon_output[6]))


if __name__ == '__main__':
    os.makedirs(outdir, exist_ok=True)
    outpath = os.path.join(outdir, outfile_name)
    print(f'Output: {outpath}')

    with h5py.File(outpath, 'w') as fout:
        g = fout.create_group('recon_histograms')
        bc = 0.5 * (hist_bins[:-1] + hist_bins[1:])
        d = g.create_dataset('bc', data=bc, dtype=np.float64)
        d.attrs['unit'] = 'keV/c'

        for gas_type, entries in datasets_config.items():
            grp = g.create_group(gas_type)
            for dataset, data_type, file_prefix, nfiles in entries:

                print(f'  {gas_type}/{dataset} ({nfiles} files)...')
                recon = read_recon_all(dataset, data_type, file_prefix, nfiles)
                _, hh, hh_nocal = get_summed_histogram(recon, hist_bins, remove_doublecounts=True)

                unit = f'count/{hist_bins[1]-hist_bins[0]:.0f}keV'
                p = mean_pressure(recon)

                d = grp.create_dataset(dataset, data=hh, dtype=np.int64)
                d.attrs['pressure_mbar'] = p
                d.attrs['unit'] = unit

                d2 = grp.create_dataset(dataset + '_nocal', data=hh_nocal, dtype=np.int64)
                d2.attrs['pressure_mbar'] = p
                d2.attrs['unit'] = unit

    print('Done.')
