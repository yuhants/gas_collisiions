import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

from scipy.signal import butter, sosfilt, iirnotch, filtfilt
from scipy.fft import rfft, irfft, rfftfreq
import h5py

from scipy.signal import welch
from scipy.optimize import curve_fit, minimize

from scipy.special import voigt_profile

c = 299792458  # m/s
SI2ev = (1 / 1.6e-19) * c

def load_plotting_setting():
    # colors=['#fe9f6d', '#de4968', '#8c2981', '#3b0f70', '#000004']
    # colors = plt.colormaps.get_cmap('tab20b').resampled(6).colors
    cmap = plt.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0.1, 0.95, 5))

    default_cycler = cycler(color=colors)
    
    params = {'figure.figsize': (7, 5),
              'axes.prop_cycle': default_cycler,
              'axes.titlesize': 14,
              'legend.fontsize': 12,
              'axes.labelsize': 14,
              'axes.titlesize': 14,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'xtick.direction': 'in',
              'ytick.direction': 'in',
              'xtick.top': True,
              'ytick.right': True
              }
    plt.rcParams.update(params)

def load_timestreams(file, channels=['C']):
    timestreams = []
    delta_t = None

    if file[-5:] == '.hdf5':
        f = h5py.File(file, 'r')
        for c in channels:
            # Convert mv to V
            adc2mv = f['data'][f'channel_{c.lower()}'].attrs['adc2mv']
            timestreams.append(f['data'][f'channel_{c.lower()}'][:] * adc2mv / 1000)

        if delta_t is None:
                delta_t = f['data'].attrs['delta_t']
        f.close()
            
    return delta_t, timestreams

#### Filtering
def notch_filtered(data, fs, f0=93000, q=50):
    b, a = iirnotch(f0, q, fs)
    filtered = filtfilt(b, a, data)
    return filtered

def bandpass_filtered(data, fs, f_low=10000, f_high=100000, order=2): 
    sos_bp = butter(order, [f_low, f_high], 'bandpass', fs=fs, output='sos')
    filtered = sosfilt(sos_bp, data)
    return filtered

def lowpass_filtered(tod, fs, f_lp=50000, order=2):
    sos_lp = butter(order, f_lp, 'lp', fs=fs, output='sos')
    filtered = sosfilt(sos_lp, tod)
    return filtered

def highpass_filtered(tod, fs, f_hp=50000, order=2):
    sos_hp = butter(order, f_hp, 'hp', fs=fs, output='sos')
    filtered = sosfilt(sos_hp, tod)
    return filtered

#### Calibration
def get_psd(dt=None, tt=None, zz=None, nperseg=None):
    if dt is not None:
        fs = int(np.round(1 / dt))
    elif tt is not None:
        fs = int(np.ceil(1 / (tt[1] - tt[0])))
    else:
        raise SyntaxError('Need to supply either `dt` or `tt`.')
    
    if nperseg is None:
        nperseg = fs / 10
    ff, pp = welch(zz, fs=fs, nperseg=nperseg)
    return ff, pp

def get_area_driven_peak(ffd, ppd, passband=(88700, 89300), noise_floor=None, plot=False):
    """Integrate power in PSD over passband"""
    if noise_floor is None:
        noise_idx = np.logical_and(ffd > 100000, ffd < 105000)
        noise_floor = np.mean(ppd[noise_idx])
    
    all_idx = np.logical_and(ffd > passband[0], ffd < passband[1])
    area_all = np.trapz(ppd[all_idx]-noise_floor, ffd[all_idx]*2*np.pi)
    v2_drive = area_all / (2 * np.pi)

    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.plot(ffd[all_idx], ppd[all_idx])
        ax.set_xlabel('Frequency (Hz)')
        ax.set_ylabel('Spectral density (V$^2$ / Hz)')
        ax.set_yscale('log')
        plt.show()

    return v2_drive

def get_c_mv(data_files_ordered, vp2p, omegad, passband, searchband=(25000, 40000), charge=3, n_chunk=10, efield=106, return_psds=False, sphere_radius=50e-9):
    m = 2000 * (sphere_radius**3) * (4 / 3) * np.pi  # sphere mass
    
    data_files_ordered = np.asarray(data_files_ordered)
    vp2p = np.asarray(vp2p)

    ffss, ppss = [], []
    for file in data_files_ordered:
        dtt, nn = load_timestreams(file, ['D'])
        zz = nn[0]

        size_per_chunk = int(zz.size / n_chunk)
        ffs, pps = [], []

        for i in range(n_chunk):
            ff, pp = get_psd(dt=dtt, zz=zz[i*size_per_chunk : (i+1)*size_per_chunk], nperseg=2**16)
            ffs.append(ff)
            pps.append(pp)

        ffss.append(ffs)
        ppss.append(pps)
        
    c_cals = []
    for i, vpp in enumerate(vp2p):
        fd0 = (vpp / 2) * efield * charge * 1.6e-19

        c_cal = []
        for j, ff in enumerate(ffss[i]):
            pp = ppss[i][j]
            v2_drive = get_area_driven_peak(ff, pp, passband=passband, noise_floor=None, plot=False)

            idx_band = np.logical_and(ff > searchband[0], ff < searchband[1])
            omega0 = 2 * np.pi * ff[idx_band][np.argmax(pp[idx_band])]
            z2_drive = (fd0**2 / 2) / ((m * (omega0**2 - omegad**2))**2)

            c_cal.append(v2_drive / z2_drive)
        c_cals.append(c_cal)
    
    c_mvs = np.sqrt(1 / np.asarray(c_cals))

    if return_psds:
        return c_mvs, ffss, ppss

    return c_mvs

def voigt(xx, A, x0, sigma, gamma):
    # Note that scipy.special.voigt_profile is already normalized
    return A * voigt_profile(xx-x0, sigma, gamma)

def fit_sigma_voigt(ff, sz, fit_band=(28000, 33000), gamma=None, p0=None, peak_half_width=1000):
    if p0 is None:
        p0 = [7e-18, 46147*2*np.pi, 50*2*np.pi, 5*2*np.pi]

    # Detect peak in the broad fit_band, then restrict fit to a narrow window
    # around it so the peak bins dominate the sigma=sz weighted chi-squared
    coarse_idx = np.logical_and(ff > fit_band[0], ff < fit_band[1])
    peak_freq = ff[coarse_idx][np.argmax(sz[coarse_idx])]
    fit_idx_voigt = np.logical_and(ff > peak_freq - peak_half_width,
                                   ff < peak_freq + peak_half_width)

    # Use detected peak as initial frequency guess
    p0 = list(p0)
    p0[1] = peak_freq * 2 * np.pi

    bounds = ([0, fit_band[0]*2*np.pi, 0, 0],
              [np.inf, fit_band[1]*2*np.pi, peak_half_width*2*np.pi, peak_half_width*2*np.pi])
    popt, pcov = curve_fit(lambda xx, A, f0, sigma, gamma: voigt(xx, A, f0, sigma, gamma), ff[fit_idx_voigt]*2*np.pi, sz[fit_idx_voigt],
                           p0=p0, sigma=sz[fit_idx_voigt], bounds=bounds, maxfev=50000)
    # print(popt)

    # If don't want to fit for gamma
    # popt, pcov = curve_fit(lambda xx, A, f0, sigma: voigt(xx, A, f0, sigma, gamma), ff[fit_idx_voigt]*2*np.pi, sz[fit_idx_voigt], 
    #                        p0=[1e-18, 50000*2*np.pi, 43*2*np.pi], sigma=sz[fit_idx_voigt])
    
    return popt

def gauss(x, A, mu, sigma):
    return A * np.exp(-(x-mu)**2/(2*sigma**2))

def gauss_zero(x, A, sigma):
    return A * np.exp(-(x)**2/(2*sigma**2))

def gaussian_convolved_lineshape(omega, A, omega0, sigma, gamma):
    xx_gauss = np.arange(-400*2*np.pi, 400*2*np.pi, 1)
    gauss_kernel = gauss(xx_gauss, A=1, mu=0, sigma=sigma)
    gauss_kernel /= np.sum(gauss_kernel)

    xx = np.arange(10000*2*np.pi, 120000*2*np.pi, 1)
    actual_lineshape = A / ((xx**2 - omega0**2)**2 + gamma**2 * xx**2)
    
    convolved_lineshape = np.convolve(gauss_kernel, actual_lineshape, 'same')
    return np.interp(omega, xx, convolved_lineshape)

def get_effective_force_noise(_file, c_mv, int_band=(20000, 50000), fit_band=(29000, 32000), nperseg=2**17, plot_fit=False, p0=None, sphere_radius=50e-9):
    m = 2000 * (sphere_radius**3) * (4 / 3) * np.pi  # sphere mass

    dtt, nn = load_timestreams(_file, ['D'])
    fs = int(np.ceil(1 / dtt))
    zz = nn[0]

    ff, pp = welch(zz, fs, nperseg=nperseg)
    sz_measured = pp * c_mv**2
    A, omega0, sigma, gamma_voigt = fit_sigma_voigt(ff, sz_measured, fit_band=fit_band, p0=p0)
    
    if plot_fit:
        fig, ax = plt.subplots(1, 1, figsize=(6, 4))
        ax.plot(ff/1000, sz_measured, 'b')
        ax.plot(ff/1000, voigt(ff*2*np.pi, A, omega0, sigma, gamma_voigt), 'r')
        ax.set_yscale('log')
        ax.set_ylim(1e-28, 1e-19)
        ax.set_xlim(10, 100)
        ax.set_xlabel('Frequency (kHz)')
        ax.set_ylabel(r'$S_z[\omega] (\mathrm{m}^2/\mathrm{Hz})$')
        # plt.plot()

    gamma = gamma_voigt * 2
    # Use the sigma and gamma extracted from voigt fit
    # to calculate the broadend lineshape
    # gamma/2 is the corresponding Lorentzian linewidth if the true oscillator
    # had damping gamma
    convolved_lineshape = gaussian_convolved_lineshape(ff*2*np.pi, 1, omega0, sigma, gamma)
    chi_2_convolved = (1 / (m**2)) * convolved_lineshape

    sf_measured_convolved = sz_measured / (chi_2_convolved)
    idx_int = np.logical_and(ff>int_band[0], ff<int_band[1])
    dp_kev = np.sqrt( 1/( np.trapz(4/sf_measured_convolved[idx_int], x=ff[idx_int]*2*np.pi) /(2*np.pi) ) ) * SI2ev / 1000

    if plot_fit:
        return ff, sz_measured, sf_measured_convolved, chi_2_convolved, dp_kev, fig, ax, voigt(ff*2*np.pi, A, omega0, sigma, gamma_voigt)
    else:
        return ff, sz_measured, sf_measured_convolved, chi_2_convolved, dp_kev
    
####
#### Impulse reconstruction
####
def get_pulse_idx(drive_sig, trigger_val=0.5, positive=True):
    if positive:
        return np.flatnonzero((drive_sig[:-1] < trigger_val) & (drive_sig[1:] > trigger_val))+1
    else:
        return np.flatnonzero((drive_sig[:-1] > trigger_val) & (drive_sig[1:] < trigger_val))+1

def get_analysis_window(dd, pulse_idx, length):
    window = np.full(dd.size, True)
    pulse_idx_in_window = length

    if length < pulse_idx:
        window[:pulse_idx-length] = False
    else:
        # Pulse happens at the beginning of the file
        # so it's not in the middle of the window
        pulse_idx_in_window = pulse_idx

    if (pulse_idx + length) < (dd.size-1):
        window[pulse_idx+length:] = False
    
    return window, pulse_idx_in_window

def get_prepulse_window(tt, pulse_idx, length):
    window = np.full(tt.size, True)
    window[:pulse_idx-int(length/2)] = False
    window[pulse_idx+int(length/2):] = False
    
    return window

def get_sv_imp(fs, zz, fit_band=(44000, 60000), noise_band=(110000, 120000),
              nperseg=2**19, p0=None):
    """Compute the Welch PSD of the z signal and fit the resonance peak.

    Fits a Voigt profile to the resonance peak within fit_band to extract
    omega0, sigma, and gamma. Also measures the imprecision noise floor sv_imp
    as the mean PSD in noise_band (an off-resonance region above the peak but
    below the notch).

    The caller is responsible for deriving c_imp from the returned parameters.
    In process_impulse_calibration.py this is done as:

        gamma_damping = gamma_voigt * 2
        c_imp = (pi / (omega0^2 * gamma_damping)) * sv_imp / A

    Parameters
    ----------
    fs : int
        Sampling rate in Hz.
    zz : array_like
        Raw (or notch-filtered) z signal in V.
    fit_band : tuple of float
        (f_low, f_high) in Hz. The peak is detected by argmax within this band,
        then the Voigt is fit over a narrow window around it (see fit_sigma_voigt).
    noise_band : tuple of float
        (f_low, f_high) in Hz of an off-resonance region used to measure sv_imp.
        Should be above the resonance but below the notch frequency.
    nperseg : int
        Welch segment length. Default 2**19 gives ~9.5 Hz resolution at 5 MHz.
    p0 : array_like or None
        Initial parameters [A, omega0_rad, sigma_rad, gamma_rad] for the Voigt
        fit. The f0 component is overridden by the detected peak frequency.

    Returns
    -------
    p_fit : list of float
        Voigt fit parameters [A, omega0, sigma, gamma] in SI units
        (A in V²/Hz·rad, omega0/sigma/gamma in rad/s).
    sv_imp : float
        Mean PSD in noise_band (V²/Hz), used as the imprecision noise floor.
    ff : ndarray
        Frequency array from Welch (Hz).
    pp : ndarray
        PSD array from Welch (V²/Hz).
    """
    ff, pp = welch(zz, fs=fs, nperseg=nperseg)
    A, omega0, sigma, gamma_voigt = fit_sigma_voigt(ff, pp, fit_band=fit_band, p0=p0)

    _idx = np.logical_and(ff>noise_band[0], ff<noise_band[1])
    sv_imp = np.mean(pp[_idx])

    # if fixed_snr is not None:
    #     snr = fixed_snr
    # else:
    #     pp_peak = np.max(voigt(ff * 2 * np.pi, A, omega0, sigma, gamma_voigt))
    #     noise_idx = np.logical_and(ff > noise_band[0], ff < noise_band[1])
    #     noise_floor = np.median(pp[noise_idx])
    #     snr = pp_peak / noise_floor

    # c_imp = np.pi / (2 * snr * omega0**2 * fixed_gamma * bw)
    return [A, omega0, sigma, gamma_voigt], sv_imp, ff, pp

def get_susceptibility(omega, omega0, gamma):
    """Functional form of susceptibility of an harmonic oscillator, normalized
    to mass=1"""
    ## Note that this is *not* how susceptibility is usually
    ## defined.
    ## DO NOT use this function for other calculations
    chi = 1 / (omega0**2 - omega**2 - 1j*gamma*omega)
    return chi

def get_pulse_amp(dt, zz, omega0, gamma, c_imp=None):
    zzk = rfft(zz)
    ff = rfftfreq(zz.size, dt)
    omega = ff * 2 * np.pi

    chi_omega = get_susceptibility(omega, omega0, gamma)
    # If no imprecision noise floor is provided
    # use simple 1/chi as the optimal filter
    if c_imp is None:
        filter_output = irfft(zzk / chi_omega)
    else:
        # Noise model for the optimal filter
        j_noise = np.abs(chi_omega)**2 + c_imp
        filter_output = irfft(zzk * np.conj(chi_omega) / j_noise)

    return filter_output

def get_search_window(amp, pulse_idx_in_window, search_window_length, pulse_length=20):
    search_window = np.full(amp.size, False)
    left  = pulse_idx_in_window + pulse_length
    right = left + search_window_length

    if right > amp.size:
        print('Skipping pulse too close to the edge of search window')
        return None

    search_window[left:right] = True
    return search_window

def recon_pulse(idx, dtt, zz_bp, dd,
                c_imp=None,
                gamma_damping=None,
                analysis_window_length=100000,
                prepulse_window_length=5000,
                search_window_length=20,
                pulse_length=20,
                lowpass_freq=60000,
                lowpass_order=3):

    if idx < prepulse_window_length:
        print('Skipping pulse too close to the beginning of the file')
        return None, None, None, np.nan

    fs = int(np.ceil(1 / dtt))

    window, pulse_idx_in_window = get_analysis_window(dd, idx, analysis_window_length)
    prepulse_window = get_prepulse_window(dd, idx, prepulse_window_length)

    # FFT the bandpassed z signal in the prepulse window to find
    # the resonant frequency
    zzk = rfft(zz_bp[prepulse_window])
    ff = rfftfreq(zz_bp[prepulse_window].size, dtt)
    pp = np.abs(zzk)**2 / (zz_bp[prepulse_window].size / dtt)

    # Now just take the max fft frequency as the resonant frequency
    omega0_guess = ff[np.argmax(pp)] * 2 * np.pi
    omega0_fit = omega0_guess

    # Use a fixed damping to reconstruct pulse amp
    # Actual damping doesn't matter as long as gamma << omega0
    # If damping is not provided, use the frequency resolution set by FFT
    if gamma_damping is None:
        gamma_damping = 2 * np.pi * (ff[1]-ff[0])
    amp = get_pulse_amp(dtt, zz_bp[window], omega0_fit, gamma_damping, c_imp)

    # Low pass the reconstructed amplitude to reject high frequency noise
    amp_lp = lowpass_filtered(amp, fs, lowpass_freq, lowpass_order)

    # Search the absolute value because homodyne could lock differently from time to time
    search_window = get_search_window(amp, pulse_idx_in_window, search_window_length, pulse_length)
    recon_amp = np.max(np.abs(amp_lp[search_window])/1e9)

    return window, amp/1e9, amp_lp/1e9, recon_amp

def fit_amps_gaus(normalized_amps, bins=None, noise=False, return_bins=False):
    hhs, bcs, gps = [], [], []
    bins_ret = []
    for amp in normalized_amps:
        if amp.size == 0:
            continue
        if bins is None:
            bin = np.linspace(0, np.max(amp)*1.5, 50)
        else:
            bin = bins
        hh, be = np.histogram(amp, bins=bin)
        bc = 0.5 * (be[1:] + be[:-1])
        
        if noise:
            gp, gcov = curve_fit(gauss_zero, bc, hh, p0=[np.max(hh), np.std(np.abs(amp))], bounds=([1e-10, np.inf]), maxfev=100000)
        else:
            gp, gcov = curve_fit(gauss, bc, hh, p0=[np.max(hh), np.mean(np.abs(amp)), np.std(np.abs(amp))], bounds=([1e-10, np.inf]), maxfev=50000)

        hhs.append(hh)
        bcs.append(bc)
        gps.append(gp)
        bins_ret.append(bin)
    
    if return_bins:
        return hhs, bcs, gps, bins_ret
    else:
        return hhs, bcs, gps

def get_unnormalized_amps(data_files, 
                          noise=False,
                          no_search=False,
                          positive_pulse=True,
                          notch_freq=119000,
                          passband=(30000, 80000),
                          analysis_window_length=50000,
                          prepulse_window_length=50000,
                          search_window_length=250,
                          search_offset_length=20,
                          lowpass_freq=80000,
                          lowpass_order=2
                          ):
    amps = []
    for file in data_files:
        dtt, nn = load_timestreams(file, ['D', 'G'])
        fs = int(np.ceil(1 / dtt))
        zz, dd = nn[0], nn[1]

        if notch_freq is not None:
            zz = notch_filtered(zz, fs, f0=notch_freq, q=50)

        bandpass_lb, bandpass_ub = passband
        zz_bp = bandpass_filtered(zz, fs, bandpass_lb, bandpass_ub, lowpass_order)

        trigger_level = positive_pulse * 0.5
        pulse_idx = get_pulse_idx(dd, trigger_level, positive_pulse)
        
        if noise:
            # Fit noise away from the pulses
            pulse_idx = np.ceil(0.5 * (pulse_idx[:-1] + pulse_idx[1:])).astype(np.int64)

        for i, idx in enumerate(pulse_idx):
            if idx < prepulse_window_length:
                print('Skipping pulse too close to the beginning of file')
                continue
            if idx > (zz.size - analysis_window_length):
                print('Skipping pulse too close to the end of file')
                continue

            # 20241205: use a much narrower search window (25 indices; 5 us)
            # 20250211: change window length to 50000 indices and search window to 50 us
            # to be consistent with DM search
            window, f, f_lp, amp = recon_pulse(idx, dtt, zz_bp, dd, 
                                               analysis_window_length, 
                                               prepulse_window_length, 
                                               search_window_length, 
                                               search_offset_length,
                                               lowpass_freq,
                                               lowpass_order)

            if noise:
                if np.isnan(amp):
                    pass
                elif no_search:
                    # If no search, just take th middle value
                    amps.append(np.abs(f_lp[np.ceil(f_lp.size/2).astype(np.int64)])/1e9)
                else:
                    amps.append(amp)
            else:
                amps.append(amp)

    amps = np.asarray(amps)
    return amps

def get_pulse_times(data_files, positive_pulse, prepulse_window_length, analysis_window_length):
    pulse_indices, pulse_times = [], []

    for file in data_files:
        with h5py.File(file, 'r') as f:
            timestamp = f['data'].attrs['timestamp']

        dtt, nn = load_timestreams(file, ['G'])
        dd = nn[0]

        trigger_level = positive_pulse * 0.5
        pulse_idx = get_pulse_idx(dd, trigger_level, positive_pulse)

        good_idx = np.logical_and(pulse_idx > prepulse_window_length, pulse_idx<(dd.size-analysis_window_length))
        good_pulse_idx = pulse_idx[good_idx]

        pulse_indices.append(good_pulse_idx)
        pulse_times.append(timestamp + dtt*good_pulse_idx)
    
    return np.concatenate(pulse_indices), np.concatenate(pulse_times)

def recon_force(dtt, zz_bp, f_lp, lowpass_order):
    fs = int(np.ceil(1 / dtt))

    zzk = rfft(zz_bp)
    ff = rfftfreq(zz_bp.size, dtt)
    pp = np.abs(zzk)**2 / (zz_bp.size / dtt)

    omega0_fit = ff[np.argmax(pp)] * 2 * np.pi
    amp = get_pulse_amp(dtt, zz_bp, omega0_fit, (ff[1]-ff[0])*2*np.pi)    
    amp_lp = lowpass_filtered(amp, fs, f_lp, lowpass_order)

    return amp/1e9, amp_lp/1e9