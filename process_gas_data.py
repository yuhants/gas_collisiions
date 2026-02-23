import numpy as np
import os
import h5py

from scipy.signal import decimate
import analysis_utils as utils

sphere = 'sphere_20260215'

# ============================================================
# CONFIG — edit these before running
# ============================================================

# Bandpass filter bounds (Hz)
# bandpass_lb, bandpass_ub = (35000, 70000)  # sphere_20251212
# bandpass_lb, bandpass_ub = (39000, 74000)  # sphere_20260105
bandpass_lb, bandpass_ub = (38000, 75000)    # sphere_20260215

# Voltage-to-energy calibration factor
# amp2kev = 6792.86423779262  # sphere_20260105
amp2kev = 8363.560351624732   # sphere_20260215

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
        ('20260219_p6e_4e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_4e-7mbar_1', 'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_2e-7mbar',   'xenon_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar',   'xenon_data', '20260219_dfg_', 25),
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
        ('20260219_p6e_3e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_3e-7mbar_1', 'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_1e-7mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_7e-8mbar',   'sf6_data', '20260219_dfg_', 25),
        ('20260219_p6e_5e-8mbar',   'sf6_data', '20260219_dfg_', 25),
    ],
}
# ============================================================

# Impulse reconstruction filter settting
lowpass_order = 3
notch_freq = 137000

# Params for identifying pulse indices
positive_pulse = True
trigger_val = 0.5 * positive_pulse

analysis_window_length = 2**19    # Length of analysis window in number of indices
search_window_length   = 2**8     # 50 us search window

# For calculating chi2, we simply assume an approximate 60 keV sigma
sigma_p_amp = 60 / amp2kev

def get_idx_in_window(amp_searched_idx, search_window_length, lb):
    ret = np.empty_like(amp_searched_idx)

    for i, amp_idx in enumerate(amp_searched_idx):
        ret[i] = amp_idx + lb + search_window_length * i
    
    return ret

def bad_detection_quality(zz_windowed, zz_bp_windowed):
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
    pulse_shape_file = np.load(rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/pulse_calibration/{sphere}_pulse_shape_template_combined.npz')
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
    out_dir = rf'/Users/yuhan/work/nanospheres/data/gas_data_processed/{sphere}/{type}/{dataset}'
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    normalized_template = get_normalized_template(sphere, bounds=(1250, 1750), downsampled=False)

    for i in range(nfile):
        outfile_name = f'{data_prefix}{i+idx_start}_processed.hdf5'

        file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
        f = h5py.File(file, "r")

        dtt = f['data'].attrs['delta_t']
        fs = int(np.ceil(1 / dtt))   # Sampling rate at Hz
        zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V
        
        # Identify the position of applied impulses
        gg = f['data']['channel_g'][:] * f['data']['channel_g'].attrs['adc2mv'] / 1e3  # Signal in V
        pulse_indices = utils.get_pulse_idx(gg, trigger_val, positive_pulse)

        zz_notched = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
        zz_bp = utils.bandpass_filtered(zz_notched, fs, bandpass_lb, bandpass_ub, order=lowpass_order)

        zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / analysis_window_length), analysis_window_length))
        zz_shaped = np.reshape(zz, (int(zz.size / analysis_window_length), analysis_window_length))

        # Minus 3 because trowing away 2/1 amplitudes at the beginning/end of the analysis window
        amp_all         = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length/search_window_length)-3), dtype=np.float64)
        idx_in_window   = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length/search_window_length)-3), dtype=np.int32)
        good_detection  = np.full(shape=zz_bp_shaped.shape[0], fill_value=True)
        chisquare       = np.empty_like(amp_all)
        noise_level_amp = np.empty(shape=zz_bp_shaped.shape[0])
        f_res           = np.empty(shape=zz_bp_shaped.shape[0])
        drive_area      = np.empty(shape=zz_bp_shaped.shape[0])

        lb, ub = 2 * search_window_length, -1 * search_window_length
        for j, _zz_bp in enumerate(zz_bp_shaped):
            amp, amp_lp = utils.recon_force(dtt, _zz_bp, bandpass_ub, lowpass_order)

            # Throw away the beginning and the end of the reconstructed amplitudes
            # to avoid windowing effects
            amp_search = np.abs(amp_lp[lb:ub])
            amp_reshaped = np.reshape(amp_search, (int(amp_search.size/search_window_length), search_window_length))

            amp_searched_idx = np.argmax(amp_reshaped, axis=1)
            amp_searched_idx_in_window = get_idx_in_window(amp_searched_idx, search_window_length, lb)
            amp_all[j] = amp_lp[amp_searched_idx_in_window]
            idx_in_window[j] = amp_searched_idx_in_window

            # Calculate chi2 for each amplitude
            chisquare[j] = calc_chisquares(amp_lp, amp_searched_idx_in_window, normalized_template, sigma_amp=sigma_p_amp)

            # Noise level in amplitude in the time window
            noise_level_amp[j] = np.std(amp_lp[lb:ub])

            # Identify period of poor detection quality
            if bad_detection_quality(zz_shaped[j], zz_bp_shaped[j]):
                good_detection[j] = False

            # Caculate the power of the driven tone
            f_res[j], drive_area[j] = get_driven_power(dtt, zz_shaped[j], notch_freq)

        with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
            print(f'Writing file {os.path.join(out_dir, outfile_name)}')

            g = fout.create_group('data_processed')
            g.attrs['pressure_mbar'] = f['data'].attrs['pressure_mbar']
            g.attrs['timestamp'] = f['data'].attrs['timestamp']

            g.create_dataset('pulse_indices', data=pulse_indices, dtype=np.int32)
            g.create_dataset('amplitude', data=amp_all, dtype=np.float64)
            g.create_dataset('idx_in_window', data=idx_in_window, dtype=np.int32)

            g.create_dataset('good_detection', data=good_detection, dtype=np.bool_)
            g.create_dataset('noise_level_amp', data=noise_level_amp, dtype=np.float64)
            g.create_dataset('f_res', data=f_res, dtype=np.float64)
            g.create_dataset('driven_power', data=drive_area, dtype=np.float64)
            g.create_dataset('chisquare', data=chisquare, dtype=np.float64)

            fout.close()

        f.close()

if __name__ == '__main__':
    import sys
    # Optionally filter to specific gas types:
    #   python process_gas_data.py xenon sf6
    # Run all gas types if no argument is given.
    gas_filter = set(sys.argv[1:]) if len(sys.argv) > 1 else set(datasets_config)

    for gas_type, entries in datasets_config.items():
        if gas_type not in gas_filter:
            continue
        for dataset, data_type, file_prefix, nfiles in entries:
            print(f'\n[{gas_type}] {dataset}')
            process_dataset(sphere, dataset, data_type, file_prefix, nfiles, idx_start=0)