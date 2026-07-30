import numpy as np
import os, glob
import h5py

import analysis_utils as utils

# ---------------------------------------------------------------------------
# Configuration — edit these before running
# ---------------------------------------------------------------------------

sphere = 'sphere_20260215'

datasets = [
    '20260219_p6e_4e-8mbar_d137khz_0',
    '20260219_p6e_4e-8mbar_d137khz_1_afterxe',
    '20260219_p6e_4e-8mbar_d137khz_2_afterkr',
    '20260219_p6e_4e-8mbar_d137khz_3_aftersf6',
]

# sphere = 'sphere_20260105'
# datasets = [
#     '20260107_p8e_4e-8mbar_d137khz_0',
#     '20260107_p8e_4e-8mbar_d137khz_1',
#     '20260107_p8e_4e-8mbar_d137khz_2',
#     '20260107_p8e_4e-8mbar_d137khz_3',
# ]

# sphere = 'sphere_20251212'
# datasets = [
#     '20251215_p8e_5e-8mbar_d137khz_0',
#     '20251215_p8e_5e-8mbar_d137khz_1',
# ]

data_prefix = '20260219_dfg_p6e_200ns_'
# data_prefix = '20260107_dfg_p8e_200ns_'
# data_prefix = '20251215_dfg_p8e_200ns_'

# voltages = [20]
voltages = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

data_root = rf'/Volumes/LaCie/gas_collisions/pulse_calibration/{sphere}'
out_dir   = rf'/Users/yuhan/work/nanospheres/gas_collisions/data_processed/pulse_calibration/{sphere}'

## Analysis settings
# bandpass_lb, bandpass_ub = (35000, 70000) # Analysis bandwidth in Hz (Sphere 20251212)
# bandpass_lb, bandpass_ub = (39000, 74000) # Analysis bandwidth in Hz (Sphere 20260105)
# bandpass_lb, bandpass_ub = (38000, 75000)   # Analysis bandwidth in Hz (Sphere 20260215)
bandpass_lb, bandpass_ub = (35000, 80000)   # Analysis bandwidth in Hz (Sphere 20260215; after introducing imprecision)
lowpass_order  = 3

# An arbitrary scaling parameter for the noise floor in
# the noise model; used to tune the timing/amplitude resolution trade-off
c_imp_scaling = 1/3

# Fixed reconstruction parameters. Set to a float to pin to a constant for all
# files instead of using the per-file Voigt fit result.
# Set to None to use the per-file fit (default behaviour).
fixed_gamma_damping = 1 * 2 * np.pi   # rad/s, e.g. 200 * 2 * np.pi
fixed_c_imp         = 1.5e-22         # raw units; bypasses c_imp_voigt × c_imp_scaling entirely

positive_pulse = True
notch_freq     = 137000

fit_window_length      = 2**19   # Window length to fit for frequencies
analysis_window_length = 2**18   # Length of analysis window in number of indices
search_window_length   = 2**8

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def get_pulse_shape(zz_bp_in_window, f_lp, amp, length=1500, is_scaled=False):
    if not is_scaled:
        f_lp_scaled = f_lp / 1e9
    else:
        f_lp_scaled = f_lp
    pulse_idx_in_win = np.argmin(np.abs(np.abs(f_lp_scaled) - amp))

    if f_lp_scaled[pulse_idx_in_win] > 0:
        polarity = 1
    else:
        polarity = -1

    ret    = polarity * f_lp_scaled[pulse_idx_in_win - length : pulse_idx_in_win + length]
    zz_ret = zz_bp_in_window[pulse_idx_in_win - length : pulse_idx_in_win + length]

    return zz_ret, ret, pulse_idx_in_win

def get_drive_area(idx, window_length, zz, dd, dtt, drive_freq):
    window     = utils.get_prepulse_window(dd, idx, window_length)
    zz_windowed = zz[window]

    ff, pp     = utils.get_psd(dt=dtt, zz=zz_windowed, nperseg=2**16)
    noise_idx  = np.logical_and(ff > 150000, ff < 175000)
    noise_floor = np.mean(pp[noise_idx])

    search_idx = np.logical_and(ff > 30000, ff < 60000)
    f_res      = ff[search_idx][np.argmax(pp[search_idx])]

    drive_area = utils.get_area_driven_peak(ff, pp, passband=(drive_freq-100, drive_freq+100),
                                            noise_floor=noise_floor, plot=False)
    return f_res, drive_area

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

if __name__ == '__main__':

    for dataset in datasets:
        data_folder = os.path.join(data_root, dataset)
        outfile     = f'{dataset}_processed.hdf5'
        print(f'\n=== Processing {dataset} ===')

        with h5py.File(os.path.join(out_dir, outfile), 'w') as fout:
            g = fout.create_group('data_processed')

            if fixed_c_imp is not None:
                g.attrs['fixed_c_imp'] = fixed_c_imp
            else:
                g.attrs['c_imp_scaling'] = c_imp_scaling
            if fixed_gamma_damping is not None:
                g.attrs['fixed_gamma_damping'] = fixed_gamma_damping

            for v in voltages:
                print(f'  {v}v')
                combined_path = os.path.join(data_folder, f'{data_prefix}{v}v*.hdf5')
                data_files    = sorted(glob.glob(combined_path))

                zz_pulses, pulse_shapes, amps, pulse_indices_in_win = [], [], [], []
                fs_res, drive_areas, noise_levels = [], [], []
                sv_imps, voigt_params, sv_zs = [], [], []
                ffz_saved = None

                if v == 2.5:
                    amps_noise, amps_noise_search = [], []
                    noise_drive_areas, noise_f_res, noise_noise_levels = [], [], []
                    noise_waveforms, noise_waveforms_search = [], []

                for data_file in data_files:
                    dtt, nn = utils.load_timestreams(data_file, ['D', 'G'])
                    fs = int(np.ceil(1 / dtt))
                    zz, dd = nn[0], nn[1]

                    zz_notched = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
                    zz_bp      = utils.bandpass_filtered(zz_notched, fs, bandpass_lb, bandpass_ub, order=lowpass_order)

                    if fixed_gamma_damping is not None and fixed_c_imp is not None:
                        gamma_damping = fixed_gamma_damping
                        c_imp = fixed_c_imp
                    else:
                        # `utils.get_sv_imp()` does the following:
                        #  - calculate the psd using welch
                        #  - fit a voigt profile to the observed displacement noise
                        #  - derive imprecision noise floor (in units of V^2/Hz)
                        # gamma/2 is the corresponding Lorentzian linewidth if the oscillator
                        # has damping rate gamma
                        p_fit, sv_imp, ffz, sv_z = utils.get_sv_imp(
                            fs, zz, fit_band=(44000, 60000), noise_band=(110000, 120000),
                            nperseg=2**19, p0=[5e-3, 48381*2*np.pi, 100*2*np.pi, 1*2*np.pi])
                        A, omega0, sigma, gamma_voigt = p_fit
                        gamma_damping = gamma_voigt * 2

                        # Normalize the imprecision noise to chi2 * m
                        # this will be used as the noise floor for impulse reconstruction
                        c_imp = (np.pi / (omega0**2 * gamma_damping)) * sv_imp / (A)
                        c_imp *= c_imp_scaling

                        if fixed_gamma_damping is not None:
                            gamma_damping = fixed_gamma_damping
                        if fixed_c_imp is not None:
                            c_imp = fixed_c_imp

                        sv_imps.append(sv_imp)
                        voigt_params.append(p_fit)
                        sv_zs.append(sv_z)
                        if ffz_saved is None:
                            ffz_saved = ffz

                    # Extract the pulse position
                    trigger_level = 0.5 * positive_pulse
                    pulse_indices = utils.get_pulse_idx(dd, trigger_level, positive_pulse)

                    for pulse_idx in pulse_indices:
                        window, _, f_lp, amp = utils.recon_pulse(
                            pulse_idx, dtt, zz_bp, dd, c_imp, gamma_damping,
                            analysis_window_length, fit_window_length,
                            search_window_length, 20, bandpass_ub, lowpass_order)
                        if window is None:
                            continue

                        f_res, drive_area = get_drive_area(pulse_idx, fit_window_length, zz, dd, dtt, notch_freq)

                        # If the amplitude has already been scaled by 1e9, as is now implemented,
                        # then set ``is_scaled`` to True
                        zz_pulse, pulse_shape, pulse_idx_in_win = get_pulse_shape(
                            zz_bp[window], f_lp, amp, 1500, is_scaled=True)

                        if pulse_shape.size != 3000:
                            print('Skipping pulse near the end of file')
                            continue

                        # Noise level: std of f_lp excluding edges to avoid windowing effects
                        # (same lb/ub convention as process_gas_data.py)
                        _lb = 2 * search_window_length
                        _ub = -1 * search_window_length
                        noise_level = np.std(f_lp[_lb:_ub])

                        zz_pulses.append(zz_pulse)
                        pulse_shapes.append(pulse_shape)
                        amps.append(amp)
                        pulse_indices_in_win.append(pulse_idx_in_win)
                        drive_areas.append(drive_area)
                        fs_res.append(f_res)
                        noise_levels.append(noise_level)

                    if v == 2.5:
                        noise_indices = np.ceil(0.5 * (pulse_indices[:-1] + pulse_indices[1:])).astype(np.int64)
                        for noise_idx in noise_indices:
                            window, _, f_lp, amp = utils.recon_pulse(
                                noise_idx, dtt, zz_bp, dd, c_imp, gamma_damping,
                                analysis_window_length, fit_window_length,
                                search_window_length, 20, bandpass_ub, lowpass_order)
                            if window is None:
                                continue

                            # Update 20260107: remove the divided by 1e9 scaling
                            mid = int(np.ceil(f_lp.size / 2))

                            # Waveform centred on mid (no search)
                            lo, hi = mid - 1500, mid + 1500

                            # Waveform centred on the searched peak location (mirrors get_pulse_shape)
                            search_idx = np.argmin(np.abs(np.abs(f_lp) - amp))
                            lo_s, hi_s = search_idx - 1500, search_idx + 1500

                            # Only append if both waveform slices are within bounds
                            if lo < 0 or hi > f_lp.size or lo_s < 0 or hi_s > f_lp.size:
                                continue

                            amps_noise.append(np.abs(f_lp[mid]))
                            amps_noise_search.append(amp)
                            noise_waveforms.append(f_lp[lo:hi])
                            noise_waveforms_search.append(f_lp[lo_s:hi_s])

                            _lb = 2 * search_window_length
                            _ub = -1 * search_window_length
                            noise_noise_levels.append(np.std(f_lp[_lb:_ub]))
                            _f_res, _drive_area = get_drive_area(noise_idx, fit_window_length, zz, dd, dtt, notch_freq)
                            noise_f_res.append(_f_res)
                            noise_drive_areas.append(_drive_area)

                g.create_dataset(f'amplitudes_{v}v',          data=np.asarray(amps),                dtype=np.float64)
                g.create_dataset(f'pulse_shapes_{v}v',        data=np.asarray(pulse_shapes),        dtype=np.float64)
                g.create_dataset(f'pulse_indices_in_win_{v}v',data=np.asarray(pulse_indices_in_win),dtype=np.int32)
                g.create_dataset(f'z_signal_{v}v',            data=np.asarray(zz_pulses),           dtype=np.float64)
                g.create_dataset(f'drive_area_{v}v',          data=np.asarray(drive_areas),         dtype=np.float64)
                g.create_dataset(f'f_res_{v}v',               data=np.asarray(fs_res),              dtype=np.float64)
                g.create_dataset(f'noise_level_{v}v',         data=np.asarray(noise_levels),        dtype=np.float64)
                if sv_imps:
                    g.create_dataset(f'sv_imp_{v}v',          data=np.asarray(sv_imps),             dtype=np.float64)
                    g.create_dataset(f'voigt_params_{v}v',    data=np.asarray(voigt_params),        dtype=np.float64)
                    g.create_dataset(f'sv_z_{v}v',            data=np.asarray(sv_zs),               dtype=np.float64)
                    if v == voltages[0] and ffz_saved is not None:
                        g.create_dataset('ffz', data=ffz_saved, dtype=np.float64)

                if v == 2.5:
                    g.create_dataset(f'amplitudes_noise_{v}v',        data=np.asarray(amps_noise),        dtype=np.float64)
                    g.create_dataset(f'amplitudes_noise_search_{v}v', data=np.asarray(amps_noise_search), dtype=np.float64)
                    g.create_dataset(f'noise_drive_area_{v}v',        data=np.asarray(noise_drive_areas), dtype=np.float64)
                    g.create_dataset(f'noise_f_res_{v}v',             data=np.asarray(noise_f_res),       dtype=np.float64)
                    g.create_dataset(f'noise_noise_level_{v}v',       data=np.asarray(noise_noise_levels),dtype=np.float64)
                    g.create_dataset(f'noise_waveforms_{v}v',        data=np.asarray(noise_waveforms),        dtype=np.float64)
                    g.create_dataset(f'noise_waveforms_search_{v}v', data=np.asarray(noise_waveforms_search), dtype=np.float64)

            print(f'  -> wrote {outfile}')
