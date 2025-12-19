import numpy as np
import os
import h5py

import analysis_utils as utils


sphere = 'sphere_20251129'

window_length = 50000  # 10 ms analysis window, assume dt=200 ns
search_window_length = 250  # 50 us search window

dataset = '20251201_p8e_7e-8mbar'
data_prefix = '20251201_df_'

def bad_detection_quality(zz_windowed, zz_bp_windowed):
    # Z signal out of balance, meaning that homodyne losing lock
    if np.abs(np.mean(zz_windowed)) > 0.25:
        return True
    
    if np.max(np.abs(zz_windowed)) > 0.95:
        return True

    # Check the sum over 10 indices to see if there
    # is a consecutive period of very small signal after bandpass
    convolved = np.convolve(np.abs(zz_bp_windowed),np.ones(10, dtype=int), 'valid')
    if np.sum(convolved < 1e-3) > 0:
        return True

def process_dataset(sphere, dataset, data_prefix, nfile, idx_start):
    data_dir = rf'/Volumes/LaCie/gas_collisions/background_data/{sphere}/{dataset}'
    out_dir = rf'/Users/yuhan/work/nanospheres/data/gas_data_processed/{sphere}/{dataset}'

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    file = os.path.join(data_dir, f'{data_prefix}{i+idx_start}.hdf5')
    print(file)

    f = h5py.File(file, "r")

    dtt = f['data'].attrs['delta_t']
    fs = int(np.ceil(1 / dtt))   # Sampling rate at Hz
    zz = f['data']['channel_d'][:] * f['data']['channel_d'].attrs['adc2mv'] / 1e3  # Signal in V