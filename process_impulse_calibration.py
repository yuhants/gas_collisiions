import numpy as np
import os, glob
import h5py

from scipy.signal import butter, sosfilt, iirnotch, filtfilt
from scipy.fft import rfft, irfft, rfftfreq
import analysis_utils as utils

import matplotlib.pyplot as plt

# sphere = 'sphere_20251212'
# dataset = '20251215_p8e_5e-8mbar_d137khz_1'
# data_prefix = '20251215_dfg_p8e_200ns_'

sphere = 'sphere_20260105'
dataset = '20260107_p8e_4e-8mbar_d137khz_3'
data_prefix = '20260107_dfg_p8e_200ns_'

# voltages = [20]
voltages = [2.5, 5, 7.5, 10, 12.5, 15, 17.5, 20]

data_folder = rf'/Volumes/LaCie/gas_collisions/pulse_calibration/{sphere}/{dataset}'

out_dir = r'/Users/yuhan/work/nanospheres/gas_collisiions/data_processed/pulse_calibration'
outfile = f'{dataset}_processed.hdf5'

## Analysis settings
# bandpass_lb, bandpass_ub = (35000, 70000) # Analysis bandwidth in Hz (Sphere 20251212)
bandpass_lb, bandpass_ub = (39000, 74000) # Analysis bandwidth in Hz (Sphere 20260105)
lowpass_order = 3
positive_pulse = True
notch_freq = 137000

fit_window_length = 2**19                 # Window length to fit for frequencies
analysis_window_length = 2**18            # Length of analysis window in number of indices

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

    ret = polarity * f_lp_scaled[pulse_idx_in_win - length : pulse_idx_in_win + length]
    zz_ret = zz_bp_in_window[pulse_idx_in_win - length : pulse_idx_in_win + length]

    # Get 50 us around the maximum amplitude
    return zz_ret, ret

def get_drive_area(idx, window_length, zz, drive_freq):
    window = utils.get_prepulse_window(dd, idx, window_length)
    zz_windowed = zz[window]

    ff, pp = utils.get_psd(dt=dtt, zz=zz_windowed, nperseg=2**16)
    noise_idx = np.logical_and(ff > 150000, ff < 175000)
    noise_floor = np.mean(pp[noise_idx])

    search_idx = np.logical_and(ff > 30000, ff < 60000)
    f_res = ff[search_idx][np.argmax(pp[search_idx])]

    drive_area = utils.get_area_driven_peak(ff, pp, passband=(drive_freq-100, drive_freq+100), noise_floor=noise_floor, plot=False)
    return f_res, drive_area

if __name__ == '__main__':

    with h5py.File(os.path.join(out_dir, outfile), 'w') as fout:
        g = fout.create_group('data_processed')

        for v in voltages:
            print(f'{v}v')
            combined_path = os.path.join(data_folder, f'{data_prefix}{v}v*.hdf5')
            data_files = glob.glob(combined_path)

            zz_pulses, pulse_shapes, amps = [], [], []
            fs_res, drive_areas = [], []

            if v == 2.5:
                amps_noise, amps_noise_search = [], []

            for data_file in data_files:
                dtt, nn = utils.load_timestreams(data_file, ['D', 'G'])
                fs = int(np.ceil(1 / dtt))
                zz, dd = nn[0], nn[1]

                zz_notched = utils.notch_filtered(zz, fs, f0=notch_freq, q=50)
                # First loosely bandpass the z signal
                zz_bp = utils.bandpass_filtered(zz_notched, fs, bandpass_lb, bandpass_ub, order=lowpass_order)
                
                # Extract the pulse position
                trigger_level = 0.5 * positive_pulse
                pulse_indices = utils.get_pulse_idx(dd, trigger_level, positive_pulse)
                noise_indices = np.ceil(0.5 * (pulse_indices[:-1] + pulse_indices[1:])).astype(np.int64)

                for pulse_idx in pulse_indices:
                    window, f, f_lp, amp = utils.recon_pulse(pulse_idx, dtt, zz_bp, dd, 
                                                             analysis_window_length, 
                                                             fit_window_length, 250, 20, bandpass_ub, lowpass_order)
                    if window is None:
                        continue

                    f_res, drive_area = get_drive_area(pulse_idx, fit_window_length, zz, notch_freq)

                    # If the amplitude has already been scaled by 1e9, as is now implemented,
                    # then set ``is_scaled`` to True
                    zz_pulse, pulse_shape = get_pulse_shape(zz_bp[window], f_lp, amp, 1500, is_scaled=True)

                    if pulse_shape.size != 3000:
                        print('Skipping pulse near the end of file')
                        continue

                    zz_pulses.append(zz_pulse)
                    pulse_shapes.append(pulse_shape)
                    amps.append(amp)
                    drive_areas.append(drive_area)
                    fs_res.append(f_res)
        
                if v == 2.5:
                    for noise_idx in noise_indices:
                        window, f, f_lp, amp = utils.recon_pulse(noise_idx, dtt, zz_bp, dd, 
                                                                 analysis_window_length, 
                                                                 fit_window_length, 250, 20, bandpass_ub, lowpass_order)
                        if window is None:
                            continue
                        
                        # Update 20260107: remove the divided by 1e9 scaling
                        amps_noise.append(np.abs(f_lp[np.ceil(f_lp.size/2).astype(np.int64)]))
                        amps_noise_search.append(amp)

            g.create_dataset(f'amplitudes_{v}v', data=np.asarray(amps), dtype=np.float64)
            g.create_dataset(f'pulse_shapes_{v}v', data=np.asarray(pulse_shapes), dtype=np.float64)
            g.create_dataset(f'z_signal_{v}v', data=np.asarray(zz_pulses), dtype=np.float64)
            g.create_dataset(f'drive_area_{v}v', data=np.asarray(drive_areas), dtype=np.float64)
            g.create_dataset(f'f_res_{v}v', data=np.asarray(fs_res), dtype=np.float64)

            if v == 2.5:
                g.create_dataset(f'amplitudes_noise_{v}v', data=np.asarray(amps_noise), dtype=np.float64)
                g.create_dataset(f'amplitudes_noise_search_{v}v', data=np.asarray(amps_noise_search), dtype=np.float64)

        print(f'Writing file {outfile}')
        fout.close()