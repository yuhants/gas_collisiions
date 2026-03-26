"""
gen_signal_model_mc.py
----------------------
Monte Carlo simulation of the expected gas-collision signal spectrum after the
full search pipeline (exactly mirroring process_gas_data.py).

For each requested pressure the script:
  1. Draws Poisson-distributed collision events from the kinetic-theory spectrum.
  2. Injects them into bandlimited Gaussian noise at the sphere's measured level.
  3. Runs the same sub-window argmax search used in process_gas_data.py.
  4. Saves per-window amplitude, chi-squared, noise-level, and index arrays in
     exactly the same HDF5 format as process_gas_data.py so that
     recon_histograms.py can apply quality cuts identically.

One HDF5 file is written per pressure under data_processed/signal_model_mc/<tag>/.
Each file has the same internal structure as a sliding-window process_gas_data.py
output, so recon_histograms.py can read it without modification.

Note: f_res and driven_power are not simulated.  Nominal values that pass the
recon_histograms.py drive-power cut are stored so the output can be fed into
recon_histograms.py without --no-drive-cut.  To simulate noise-only windows,
add 0 to pressures_mbar (zero pressure → zero event rate → pure noise).

Usage
-----
Edit the configuration block below, then run::

    python gen_signal_model_mc.py
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfilt
import os
import h5py

import calc_gas_collision_spectrum as calc_gas

# ============================================================
# CONFIGURATION — edit these before running
# ============================================================

# ── Sphere / calibration ──────────────────────────────────────────────────
sphere        = 'sphere_20260215'
sphere_radius = 50e-9   # m

# Path to pulse-shape template (.npz, must contain keys 'ps_20v' and 'amp2kev').
# Set to None to auto-resolve from data_processed/pulse_calibration/.
# amp2kev is read directly from this file.
pulse_template_path = None

# ── Gas / collision theory ────────────────────────────────────────────────
# Supported gas names: 'xe', 'kr', 'sf6', 'n2', 'he', 'ar'
gas = 'xe'

T_gas    = 293   # K — ambient gas temperature
T_sensor = 293.05   # K — laser-heated sphere surface temperature
alpha    = 0.9   # thermal accommodation coefficient

# Pressures (mbar) to simulate.  Each entry becomes one HDF5 output file.
# Use 0 for a noise-only run (no signal events injected).
pressures_mbar = [0, 1e-8, 3e-8, 5e-8, 7e-8, 1e-7, 3e-7, 5e-7, 7e-7, 1e-6]

# ── MC settings ───────────────────────────────────────────────────────────
n_analysis_windows = 512   # number of full 105-ms windows to simulate per pressure
rng_seed           = 42

# ── Output ────────────────────────────────────────────────────────────────
output_dir = 'data_processed/signal_model_mc'
output_tag = 'mc_xe'   # if None, auto-generated as '{sphere}_{gas}'

# ============================================================
# GAS MASS TABLE  (atomic mass units)
# ============================================================

GAS_MASSES_AMU = {
    'he':  4.0026,
    'n2':  28.014,
    'ar':  39.948,
    'kr':  83.798,
    'xe':  131.293,
    'sf6': 146.056,   # 32.06 + 6×19.00
}

# ============================================================
# FIXED DETECTOR PARAMETERS  (must match process_gas_data.py)
# ============================================================

fs                     = 5_000_000      # Hz
dt                     = 1 / fs
analysis_window_length = 2**19          # 524 288 samples ≈ 104.9 ms
search_window_length   = 2**8           # 256 samples ≈ 51.2 µs
lb                     = 2 * search_window_length   # searchable-region start (512)
n_searches             = (analysis_window_length // search_window_length) - 3   # 2045

T_analysis_window = analysis_window_length / fs
T_search          = search_window_length   / fs

# Bandpass + lowpass filter — must match process_gas_data.py for sphere_20260215
# (updated to 35–80 kHz after introducing imprecision)
bandpass_lb   = 35_000   # Hz
bandpass_ub   = 80_000   # Hz
lowpass_order = 3

# Chi-squared noise assumption — must match process_gas_data.py:
#   ds_sigma_amp = 60 / ds_amp2kev  (hardcoded, not derived from noise_level_amp)
sigma_noise_kev = 60.0   # keV/c

# Nominal f_res and driven_power written to every output window.
# Not simulated; chosen so recon_histograms.py drive-power cut passes
# (norm_drive ≈ 9.8e-9 >> threshold 4.5e-9 for these values).
_NOMINAL_F_RES      = 50_000.0   # Hz
_NOMINAL_DRIVEN_PWR = 1e-7       # a.u.

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _resolve_template_path(sphere):
    """Return the default pulse-template npz path for a sphere."""
    return os.path.join(
        'data_processed', 'pulse_calibration',
        sphere,
        f'{sphere}_impulse_recon_combined.npz',
    )


def _load_pulse_template(path):
    """
    Load normalised pulse template and amp2kev from an impulse_recon_combined.npz.

    Returns
    -------
    ps_norm                  : (3000,) float64 — normalised template (peak = 1)
    peak_cal                 : int — index of peak within ps_norm
    normalized_template_chi2 : (500,) float64 — ±250-sample chi2 template
    amp2kev                  : float — keV/c per raw amplitude unit
    """
    cal      = np.load(path)
    ps_raw   = cal['ps_20v']
    amp2kev  = float(cal['amp2kev'])
    peak_cal = int(np.argmax(ps_raw))
    ps_norm  = ps_raw / ps_raw[peak_cal]

    half_chi2 = 250
    normalized_template_chi2 = ps_norm[peak_cal - half_chi2 : peak_cal + half_chi2].copy()

    return ps_norm, peak_cal, normalized_template_chi2, amp2kev


def _calibrate_noise(sos_bp, sos_lp, sigma_noise_kev, amp2kev):
    """
    Find the white-noise std (sigma_wn) that produces std = sigma_noise_kev / amp2kev
    after bandpass + lowpass filtering.
    """
    n_char    = 2**22
    rng_char  = np.random.default_rng(9999)
    white     = rng_char.normal(0, 1, n_char)
    lp        = sosfilt(sos_lp, sosfilt(sos_bp, white))
    std_ratio = np.std(lp[n_char // 100:])   # skip filter warm-up
    return (sigma_noise_kev / amp2kev) / std_ratio


# ── Core simulation functions ───────────────────────────────────────────────

def sample_spectrum_icdf(qq, drdqz, n_samples, rng):
    """Draw n_samples momenta (keV/c) from drdqz via inverse-CDF sampling."""
    total_rate = np.trapz(drdqz, qq)
    cdf        = cumulative_trapezoid(drdqz, qq, initial=0) / total_rate
    return np.interp(rng.uniform(0, 1, n_samples), cdf, qq)


def gen_analysis_window_noise(sigma_wn, sos_bp, sos_lp, rng, n_pad=4096):
    """
    Generate one 2^19-sample amplitude noise trace via bandpass + lowpass filtered
    white Gaussian noise — same filter chain as process_gas_data.py.
    """
    total = n_pad + analysis_window_length
    white = rng.normal(0, sigma_wn, total)
    return sosfilt(sos_lp, sosfilt(sos_bp, white))[n_pad:]


def inject_signals(amp_lp, q_kev_arr, positions, ps_norm, peak_cal, amp2kev):
    """
    Inject calibration-pulse-shaped signals into the amplitude trace (in place).

    Parameters
    ----------
    amp_lp    : (analysis_window_length,) float — modified in place
    q_kev_arr : signed true impulse amplitudes (keV/c); negative = downward kick
    positions : peak indices within amp_lp (absolute, within searchable region)
    """
    win_len    = len(amp_lp)
    n_template = len(ps_norm)
    for q, pos in zip(q_kev_arr, positions):
        scale          = q / amp2kev
        t_start_in_win = pos - peak_cal
        t_end_in_win   = t_start_in_win + n_template
        w_start = max(0, t_start_in_win)
        w_end   = min(win_len, t_end_in_win)
        t_s     = w_start - t_start_in_win
        amp_lp[w_start:w_end] += scale * ps_norm[t_s : t_s + (w_end - w_start)]


def search_and_recon(amp_lp):
    """
    Sub-window search matching process_gas_data.py exactly:
    reshape amp_lp[lb : -search_window_length] into (n_searches, 256),
    argmax |amp| per row.

    Returns
    -------
    amps          : (n_searches,) signed amplitudes at peak positions
    idx_in_window : (n_searches,) int32 absolute indices within the analysis window
    """
    ub           = analysis_window_length - search_window_length
    amp_search   = amp_lp[lb:ub]
    n_sw         = amp_search.size // search_window_length
    amp_reshaped = amp_search[:n_sw * search_window_length].reshape(n_sw, search_window_length)
    local_idx    = np.argmax(np.abs(amp_reshaped), axis=1)
    idx_in_win   = (local_idx + lb
                    + search_window_length * np.arange(n_sw, dtype=np.int32))
    return amp_lp[idx_in_win], idx_in_win


def calc_chi2_all(amp_lp, idx_in_window, normalized_template_chi2, sigma_p_amp):
    """
    Vectorised chi-squared for all sub-windows using the 500-sample template,
    matching calc_chisquares() in process_gas_data.py (window_size=250).

    Returns np.inf for edge sub-windows where the 500-sample window falls outside.
    """
    half    = len(normalized_template_chi2) // 2   # 250
    win_len = len(amp_lp)
    n       = len(idx_in_window)
    chi2    = np.full(n, np.inf)

    valid = (idx_in_window >= half) & (idx_in_window + half <= win_len)
    idx_v = idx_in_window[valid]
    if idx_v.size == 0:
        return chi2

    offsets   = np.arange(-half, half)
    waveforms = amp_lp[idx_v[:, np.newaxis] + offsets[np.newaxis, :]]
    amps_v    = amp_lp[idx_v]
    templates = amps_v[:, np.newaxis] * normalized_template_chi2[np.newaxis, :]
    chi2[valid] = np.sum(((waveforms - templates) / sigma_p_amp)**2, axis=1)
    return chi2


# ── Main MC loop ────────────────────────────────────────────────────────────

def run_mc(n_windows, events_per_window, rng,
           sos_bp, sos_lp, sigma_wn, sigma_p_amp,
           ps_norm, peak_cal, normalized_template_chi2,
           qq_kev, drdqz, amp2kev_val):
    """
    Simulate n_windows full analysis windows and return raw per-window arrays
    in the same format as process_gas_data.py (no cuts applied).

    For each window:
      1. Generate bandlimited Gaussian noise.
      2. Inject Poisson(events_per_window) signal events (random positions, ±1 signs).
      3. Search: argmax |amp| in each 256-sample sub-window.
      4. Compute chi-squared against the 500-sample pulse template.
      5. Record noise-level (std of searchable region in raw amp units).

    Parameters
    ----------
    events_per_window : Poisson mean; pass 0 for noise-only (no injection).

    Returns
    -------
    amplitude       : (n_windows, n_searches) float64 — signed raw amplitudes
    idx_in_window   : (n_windows, n_searches) int32
    noise_level_amp : (n_windows,)            float64 — std in raw amp units
    chisquare       : (n_windows, n_searches) float64
    injected_events : dict with keys:
        'q_true_kev'  : (N,) float64 — true |qz| in keV/c (unsigned)
        'positions'   : (N,) int32   — peak sample position in analysis window
        'signs'       : (N,) int8    — ±1 injection sign
        'window_idx'  : (N,) int32   — analysis window index
        Sub-window index can be derived: (positions - lb) // search_window_length
    """
    amplitude_all       = np.zeros((n_windows, n_searches), dtype=np.float64)
    idx_in_window_all   = np.zeros((n_windows, n_searches), dtype=np.int32)
    noise_level_amp_all = np.zeros(n_windows,               dtype=np.float64)
    chisquare_all       = np.zeros((n_windows, n_searches), dtype=np.float64)

    inj_q_true   = []
    inj_pos      = []
    inj_signs    = []
    inj_win_idx  = []

    for i in range(n_windows):
        amp_lp = gen_analysis_window_noise(sigma_wn, sos_bp, sos_lp, rng)

        if events_per_window > 0:
            n_events = rng.poisson(events_per_window)
            if n_events > 0:
                q_true    = sample_spectrum_icdf(qq_kev, drdqz, n_events, rng)
                signs     = rng.integers(0, 2, size=n_events) * 2 - 1
                positions = rng.integers(
                    lb, analysis_window_length - search_window_length, size=n_events
                )
                inject_signals(amp_lp, q_true * signs, positions,
                               ps_norm, peak_cal, amp2kev_val)

                inj_q_true.append(q_true)
                inj_pos.append(positions)
                inj_signs.append(signs)
                inj_win_idx.append(np.full(n_events, i, dtype=np.int32))

        amps, idx_win = search_and_recon(amp_lp)
        chi2          = calc_chi2_all(amp_lp, idx_win, normalized_template_chi2, sigma_p_amp)

        amplitude_all[i]       = amps
        idx_in_window_all[i]   = idx_win
        noise_level_amp_all[i] = np.std(amp_lp[lb : analysis_window_length - search_window_length])
        chisquare_all[i]       = chi2

    injected_events = {
        'q_true_kev':  np.concatenate(inj_q_true)   if inj_q_true else np.empty(0, dtype=np.float64),
        'positions':   np.concatenate(inj_pos)       if inj_pos   else np.empty(0, dtype=np.int32),
        'signs':       np.concatenate(inj_signs)     if inj_signs else np.empty(0, dtype=np.int8),
        'window_idx':  np.concatenate(inj_win_idx)   if inj_win_idx else np.empty(0, dtype=np.int32),
    }

    return amplitude_all, idx_in_window_all, noise_level_amp_all, chisquare_all, injected_events


# ============================================================
# MAIN
# ============================================================

def main():
    gas_key = gas.lower()
    if gas_key not in GAS_MASSES_AMU:
        raise ValueError(f"Unknown gas '{gas}'. Choose from: {list(GAS_MASSES_AMU.keys())}")
    mg_amu = GAS_MASSES_AMU[gas_key]

    tag     = output_tag or f'{sphere}_{gas_key}'
    out_dir = os.path.join(output_dir, tag)
    os.makedirs(out_dir, exist_ok=True)

    # ── Filters ───────────────────────────────────────────────────────────
    sos_bp = butter(lowpass_order, [bandpass_lb, bandpass_ub], 'bandpass', fs=fs, output='sos')
    sos_lp = butter(lowpass_order, bandpass_ub, 'lp', fs=fs, output='sos')

    # ── Pulse template + amp2kev (read from calibration file) ─────────────
    tpl_path = pulse_template_path or _resolve_template_path(sphere)
    ps_norm, peak_cal, normalized_template_chi2, amp2kev = _load_pulse_template(tpl_path)
    sigma_p_amp = sigma_noise_kev / amp2kev

    # ── Noise calibration ─────────────────────────────────────────────────
    sigma_wn = _calibrate_noise(sos_bp, sos_lp, sigma_noise_kev, amp2kev)

    # ── Theory spectrum (computed at 1e-8 mbar; scaled per pressure) ──────
    qq_kev = np.linspace(0.5, 2000, 2000)
    drdq_unit = calc_gas.dgamma_dp_tot_noneq(
        qq_kev, mg_amu, p_mbar=1e-8, alpha=alpha,
        T_gas=T_gas, T_sensor=T_sensor, sphere_radius=sphere_radius,
    )
    _, drdqz_unit   = calc_gas.get_drdqz(qq_kev, drdq_unit)
    total_rate_unit = np.trapz(drdqz_unit, qq_kev)   # Hz at 1e-8 mbar

    print(f'Sphere:              {sphere}')
    print(f'Gas:                 {gas.upper()}  ({mg_amu} amu)')
    print(f'amp2kev:             {amp2kev:.3f}  (from {tpl_path})')
    print(f'bandpass:            {bandpass_lb/1e3:.0f}–{bandpass_ub/1e3:.0f} kHz')
    print(f'sigma_noise_kev:     {sigma_noise_kev:.1f} keV/c  →  sigma_p_amp = {sigma_p_amp:.4e}')
    print(f'T_gas={T_gas} K   T_sensor={T_sensor} K   alpha={alpha}')
    print(f'Rate @ 1e-8 mbar:    {total_rate_unit:.3f} Hz')
    print(f'MC windows/pressure: {n_analysis_windows}')
    print(f'Output dir:          {out_dir}')
    print()

    # ── Per-pressure runs ─────────────────────────────────────────────────
    for p_idx, p in enumerate(pressures_mbar):
        scale      = p / 1e-8 if p > 0 else 0.0
        drdqz_p    = drdqz_unit * scale
        ev_per_win = total_rate_unit * scale * T_analysis_window

        out_path = os.path.join(out_dir, f'{gas_key}_{p:.1e}mbar_mc.hdf5')

        print(f'[{p_idx+1}/{len(pressures_mbar)}] {p:.1e} mbar  '
              f'({ev_per_win:.3f} events/window)  →  {out_path}')

        rng = np.random.default_rng(rng_seed + p_idx)
        amplitude, idx_in_window, noise_level_amp, chisquare, injected = run_mc(
            n_analysis_windows, ev_per_win, rng=rng,
            sos_bp=sos_bp, sos_lp=sos_lp, sigma_wn=sigma_wn, sigma_p_amp=sigma_p_amp,
            ps_norm=ps_norm, peak_cal=peak_cal,
            normalized_template_chi2=normalized_template_chi2,
            qq_kev=qq_kev, drdqz=drdqz_p,
            amp2kev_val=amp2kev,
        )

        n_inj = len(injected['q_true_kev'])
        print(f'  → {n_inj} signal events injected')

        with h5py.File(out_path, 'w') as f:
            g = f.create_group('data_processed')
            g.create_dataset('amplitude',         data=amplitude,       dtype=np.float64)
            g.create_dataset('idx_in_window',     data=idx_in_window,   dtype=np.int32)
            g.create_dataset('good_detection',
                             data=np.ones(n_analysis_windows, dtype=bool))
            g.create_dataset('noise_level_amp',   data=noise_level_amp, dtype=np.float64)
            g.create_dataset('chisquare',         data=chisquare,       dtype=np.float64)
            g.create_dataset('f_res',
                             data=np.full(n_analysis_windows, _NOMINAL_F_RES),
                             dtype=np.float64)
            g.create_dataset('driven_power',
                             data=np.full(n_analysis_windows, _NOMINAL_DRIVEN_PWR),
                             dtype=np.float64)
            g.create_dataset('cal_pulse_indices', data=np.empty(0, dtype=np.int32))

            # Injected signal event truth info
            ig = f.create_group('mc_injected')
            ig.create_dataset('q_true_kev',  data=injected['q_true_kev'],  dtype=np.float64)
            ig.create_dataset('positions',   data=injected['positions'],   dtype=np.int32)
            ig.create_dataset('signs',       data=injected['signs'],       dtype=np.int8)
            ig.create_dataset('window_idx',  data=injected['window_idx'],  dtype=np.int32)

            # Standard attrs matching process_gas_data.py convention
            g.attrs['pressure_mbar']       = p
            g.attrs['amp2kev']             = amp2kev
            g.attrs['fixed_c_imp']         = 1.5e-22
            g.attrs['fixed_gamma_damping'] = 1 * 2 * np.pi

            # MC-specific metadata
            g.attrs['mc_gas']              = gas_key
            g.attrs['mc_T_gas']            = T_gas
            g.attrs['mc_T_sensor']         = T_sensor
            g.attrs['mc_alpha']            = alpha
            g.attrs['mc_n_windows']        = n_analysis_windows
            g.attrs['mc_rng_seed']         = rng_seed + p_idx

    print('\nDone.')


if __name__ == '__main__':
    main()
