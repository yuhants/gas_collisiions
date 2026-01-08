import numpy as np
import os
import h5py

from scipy.signal import decimate
import analysis_utils as utils

# sphere = 'sphere_20251212'
sphere = 'sphere_20260105'

## Analysis settings
# bandpass_lb, bandpass_ub = (35000, 70000) # Analysis bandwidth in Hz (Sphere 20251212)
bandpass_lb, bandpass_ub = (39000, 74000) # Analysis bandwidth in Hz (Sphere 20260105)
lowpass_order = 3
notch_freq = 137000

analysis_window_length = 2**19    # Length of analysis window in number of indices
search_window_length   = 2**8     # 50 us search window
sigma_p_amp = 0.01304540266635899

# datasets = ['20251217_unknown_5e-8mbar_kryptonpumped', '20251216_unknown_6e-8mbar_cryopumped']
# data_prefixs = ['20251217_df_', '20251216_df_']
# types = ['background_data', 'background_data']
# nfiles = [300, 300]

# datasets = ['20251215_p8e_2e-7mbar']
# data_prefixs = ['20251215_df_']
# types = ['xenon_data']
# nfiles = [400]

# datasets = ['20251217_unknown_5e-8mbar', '20251217_unknown_7e-8mbar', '20251217_unknown_2e-7mbar']
# data_prefixs = ['20251217_df_', '20251217_df_', '20251217_df_']
# types = ['krypton_data', 'krypton_data', 'krypton_data']
# nfiles = [300, 300, 300]

datasets = ['20260107_p8e_3e-8mbar_valveclosed']
data_prefixs = ['20260107_df_']
types = ['background_data']
nfiles = [150]

# datasets = ['20260107_p8e_6e-8mbar', '20260107_p8e_8e-8mbar', '20260107_p8e_1e-7mbar', '20260107_p8e_2e-7mbar']
# data_prefixs = ['20260107_df_', '20260107_df_', '20260107_df_', '20260107_df_']
# types = ['xenon_data', 'xenon_data', 'xenon_data', 'xenon_data']
# nfiles = [150, 150, 150, 150]

def get_idx_in_window(amp_searched_idx, search_window_length, lb):
    ret = np.empty_like(amp_searched_idx)

    for i, amp_idx in enumerate(amp_searched_idx):
        ret[i] = amp_idx + lb + search_window_length * i
    
    return ret

def bad_detection_quality(zz_windowed, zz_bp_windowed):
    # Z signal out of balance, meaning that homodyne losing lock
    if np.abs(np.mean(zz_windowed)) > 0.25:
        return True
    
    if np.max(np.abs(zz_windowed)) > 0.95:
        return True

    # Check the sum over 100 indices to see if there
    # is a consecutive period of very small signal after bandpass
    convolved = np.convolve(np.abs(zz_bp_windowed),np.ones(100, dtype=int), 'valid')
    if np.sum(convolved < 1e-3) > 0:
        return True

def get_normalized_template(bounds=(1250, 1750), downsampled=False):
    pulse_shape_file = np.load(rf'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/pulse_calibration/sphere_20251212_pulse_shape_template_combined.npz')
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

def process_dataset(sphere, dataset, type, data_prefix, nfile, idx_start):
    data_dir = rf'/Volumes/LaCie/gas_collisions/{type}/{sphere}/{dataset}'
    out_dir = rf'/Users/yuhan/work/nanospheres/data/gas_data_processed/{sphere}/{type}/{dataset}'
    
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    # normalized_template = get_normalized_template(bounds=(1250, 1750), downsampled=False)

    for i in range(nfile):
        outfile_name = f'{data_prefix}{i+idx_start}_processed.hdf5'

        file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
        f = h5py.File(file, "r")

        dtt = f['data'].attrs['delta_t']
        fs = int(np.ceil(1 / dtt))   # Sampling rate at Hz
        zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V
        
        zz = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
        zz_bp = utils.bandpass_filtered(zz, fs, bandpass_lb, bandpass_ub, order=lowpass_order)

        zz_bp_shaped = np.reshape(zz_bp, (int(zz_bp.size / analysis_window_length), analysis_window_length))
        zz_shaped = np.reshape(zz, (int(zz_bp.size / analysis_window_length), analysis_window_length))

        # Minus 3 because trowing away 2/1 amplitudes at the beginning/end of the analysis window
        amp_all         = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length/search_window_length)-3), dtype=np.float64)
        idx_in_window   = np.empty(shape=(zz_bp_shaped.shape[0], int(analysis_window_length/search_window_length)-3), dtype=np.int16)
        good_detection  = np.full(shape=zz_bp_shaped.shape[0], fill_value=True)
        # chisquare       = np.empty_like(amp_all)
        noise_level_amp = np.empty(shape=zz_bp_shaped.shape[0])

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

            # Calculate chi2 for each amplitude
            # chisquare[j] = calc_chisquares(amp_lp, amp_searched_idx_in_window, normalized_template, sigma_amp=sigma_p_amp)

            # Noise level in amplitude in the time window
            noise_level_amp[j] = np.std(amp_lp[lb:ub])

            # Identify period of poor detection quality
            if bad_detection_quality(zz_shaped[j], zz_bp_shaped[j]):
                good_detection[j] = False

        with h5py.File(os.path.join(out_dir, outfile_name), 'w') as fout:
            print(f'Writing file {os.path.join(out_dir, outfile_name)}')

            g = fout.create_group('data_processed')
            g.attrs['pressure_mbar'] = f['data'].attrs['pressure_mbar']
            g.attrs['timestamp'] = f['data'].attrs['timestamp']

            g.create_dataset('amplitude', data=amp_all, dtype=np.float64)
            g.create_dataset('idx_in_window', data=idx_in_window, dtype=np.int16)

            g.create_dataset('good_detection', data=good_detection, dtype=np.bool_)
            g.create_dataset('noise_level_amp', data=noise_level_amp, dtype=np.float64)
            # g.create_dataset('chisquare', data=chisquare, dtype=np.float64)

            fout.close()

        f.close()

if __name__ == '__main__':
    for idx, dataset in enumerate(datasets):
        process_dataset(sphere, dataset, types[idx], data_prefixs[idx], nfiles[idx], idx_start=0)