"""
gen_signal_model_mc.py
----------------------
Monte Carlo simulation of the expected gas-collision signal spectrum after the
full search pipeline (exactly mirroring process_gas_data.py / recon_histograms.py).

For each requested (gas, pressure) pair the script:
  1. Draws Poisson-distributed collision events from the kinetic-theory spectrum.
  2. Injects them into bandlimited Gaussian noise at the sphere's measured level.
  3. Runs the same sub-window argmax search used in process_gas_data.py.
  4. Applies the same quality cuts (noise-level, chi-squared, double-count rejection),
     each of which can be disabled independently.
  5. Accumulates the measured amplitude histogram and normalises it to a differential
     event rate [Events / (keV/c) / s].

Output is saved to data_processed/signal_model_mc/<tag>.npz with per-pressure histograms
and the pure-noise floor, plus all configuration parameters as metadata.

Usage
-----
Edit the configuration block below, then run::

    python gen_signal_model_mc.py
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import butter, sosfilt
import os

import calc_gas_collision_spectrum as calc_gas

# ============================================================
# CONFIGURATION — edit these before running
# ============================================================

# ── Sphere / calibration ──────────────────────────────────────────────────
sphere           = 'sphere_20260215'
amp2kev          = 8363.560351624732   # keV/c per amplitude unit
sphere_radius    = 50e-9               # m
bandpass_lb      = 38_000              # Hz
bandpass_ub      = 75_000              # Hz
lowpass_order    = 3
sigma_noise_kev  = 60.0               # keV/c  (typical noise_level_amp × amp2kev)

# Path to the combined pulse-shape template (.npz, contains key 'ps_20v').
# Set to None to auto-resolve from data_processed/pulse_calibration/.
pulse_template_path = None

# ── Gas / collision theory ────────────────────────────────────────────────
# Supported gas names: 'xe', 'kr', 'sf6', 'n2', 'he', 'ar'
# (or pass mg_amu directly in gas_configs below)
gas = 'xe'

T_gas    = 293    # K  — ambient gas temperature
T_sensor = 900    # K  — laser-heated sphere surface temperature
alpha    = 0.9    # thermal accommodation coefficient

# Pressures (mbar) to simulate.  Each entry becomes one column in the output.
pressures_mbar = [1e-8, 3e-8, 1e-7, 3e-7, 1e-6]

# ── Quality cuts ──────────────────────────────────────────────────────────
# Set to False / np.inf to disable a cut.
apply_noise_cut  = True    # per-analysis-window noise-level cut
apply_chi2_cut   = True    # per-sub-window chi-squared cut
apply_dc_cut     = True    # double-count rejection

noise_threshold_kev = 200.0   # keV/c  (per-window noise-level upper limit)
chi2_threshold      = 700.0   # per-sub-window

# ── MC settings ───────────────────────────────────────────────────────────
n_analysis_windows = 512   # number of full 105-ms windows to simulate
rng_seed           = 42

# ── Output ────────────────────────────────────────────────────────────────
output_dir = 'data_processed/signal_model_mc'
output_tag = None   # if None, auto-generated from sphere + gas + cuts

# Output histogram binning (keV/c)
bins = np.arange(0, 2000, 25)

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

fs                     = 5_000_000           # Hz
dt                     = 1 / fs
analysis_window_length = 2**19               # 524 288 samples ≈ 104.9 ms
search_window_length   = 2**8                # 256 samples ≈ 51.2 µs
lb                     = 2 * search_window_length   # searchable-region start (512)
n_searches             = (analysis_window_length // search_window_length) - 3   # 2045

T_search          = search_window_length   / fs
T_analysis_window = analysis_window_length / fs

# Double-count rejection thresholds (ported from recon_histograms.py)
DC_AMP_THR_KEV          = 0
DC_OPP_SIGN_AMP_THR_KEV = 180
DC_IDX_THR              = 25
DC_OPP_SIGN_IDX_THR     = search_window_length // 3   # 85

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _resolve_template_path(sphere):
    """Return the default pulse-template path for a sphere name."""
    return os.path.join(
        'data_processed', 'pulse_calibration',
        f'{sphere}_pulse_shape_template_combined.npz'
    )


def _load_pulse_template(path, search_window_length):
    """
    Load normalised pulse template from .npz file.

    Returns
    -------
    ps_norm              : full normalised template (peak = 1)
    peak_cal             : index of peak within ps_norm
    normalized_template_chi2 : 500-sample chi-squared template (±250 around peak)
    fwhm_samps           : pulse FWHM in samples
    """
    cal_file = np.load(path)
    ps_raw   = cal_file['ps_20v']
    peak_cal = int(np.argmax(ps_raw))
    ps_norm  = ps_raw / ps_raw[peak_cal]

    half_chi2 = 250
    normalized_template_chi2 = ps_norm[peak_cal - half_chi2 : peak_cal + half_chi2].copy()

    half           = search_window_length // 2
    pulse_template = ps_norm[peak_cal - half : peak_cal + half]
    fwhm_samps     = int(np.sum(pulse_template > 0.5))

    return ps_norm, peak_cal, normalized_template_chi2, fwhm_samps


def _calibrate_noise(sos_bp, sos_lp, sigma_noise_kev, amp2kev):
    """
    Determine the white-noise std (sigma_wn) needed so that bandpass + lowpass
    filtered output has std = sigma_noise_kev / amp2kev.
    """
    n_char   = 2**22
    rng_char = np.random.default_rng(9999)
    white    = rng_char.normal(0, 1, n_char)
    bp       = sosfilt(sos_bp, white)
    lp       = sosfilt(sos_lp, bp)
    std_ratio = np.std(lp[n_char // 100:])
    sigma_amp = sigma_noise_kev / amp2kev
    return sigma_amp / std_ratio   # sigma_wn


# ── Core MC functions ──────────────────────────────────────────────────────

def throw_away_doublecounts(amplitude, good_det_noise, idx_in_window,
                             amp2kev=None,
                             amp_thr_kev=DC_AMP_THR_KEV,
                             opp_sign_amp_thr_kev=DC_OPP_SIGN_AMP_THR_KEV,
                             double_count_idx_thr=DC_IDX_THR,
                             opp_sign_idx_thr=DC_OPP_SIGN_IDX_THR):
    """
    Double-count rejection — ported verbatim from recon_histograms.py.

    Parameters
    ----------
    amplitude      : (n_windows, n_searches) float  — signed amplitudes in amp units
    good_det_noise : (n_windows,) bool
    idx_in_window  : (n_windows, n_searches) int    — absolute peak indices
    amp2kev        : calibration factor; uses module-level default if None

    Returns
    -------
    amplitude copy with double-count entries set to NaN.
    """
    if amp2kev is None:
        amp2kev = globals()['amp2kev']

    ret         = np.array(amplitude, dtype=np.float64)
    n_sw        = amplitude.shape[1]
    min_thr     = min(amp_thr_kev, opp_sign_amp_thr_kev)

    large_pulses = (
        (np.abs(ret) * amp2kev > min_thr)
        & np.tile(good_det_noise[:, np.newaxis], (1, n_sw))
    )
    abs_idx_large    = idx_in_window[large_pulses].astype(np.int64)
    idx_sep          = np.abs(np.diff(abs_idx_large, append=analysis_window_length))
    large_pulses_pos = np.argwhere(large_pulses)
    amplitude_large  = ret[large_pulses]

    same_peak   = idx_sep < double_count_idx_thr
    signs       = np.sign(amplitude_large)
    opp_sign    = np.append(signs[:-1] != signs[1:], False)
    peak_trough = opp_sign & (idx_sep < opp_sign_idx_thr)

    for i in range(len(large_pulses_pos) - 1):
        if np.isnan(amplitude_large[i]):
            continue
        if large_pulses_pos[i][0] != large_pulses_pos[i + 1][0]:
            continue
        min_pair_kev = min(abs(amplitude_large[i]), abs(amplitude_large[i + 1])) * amp2kev
        sp = same_peak[i]   and (min_pair_kev > amp_thr_kev)
        pt = peak_trough[i] and (min_pair_kev > opp_sign_amp_thr_kev)
        if not sp and not pt:
            continue
        if abs(amplitude_large[i]) < abs(amplitude_large[i + 1]):
            ret[large_pulses_pos[i][0],     large_pulses_pos[i][1]]     = np.nan
            amplitude_large[i]     = np.nan
        else:
            ret[large_pulses_pos[i + 1][0], large_pulses_pos[i + 1][1]] = np.nan
            amplitude_large[i + 1] = np.nan
    return ret


def sample_spectrum_icdf(qq, drdqz, n_samples, rng):
    """Draw n_samples momenta (keV/c) from drdqz via inverse-CDF sampling."""
    total_rate = np.trapz(drdqz, qq)
    cdf        = cumulative_trapezoid(drdqz, qq, initial=0) / total_rate
    u          = rng.uniform(0, 1, n_samples)
    return np.interp(u, cdf, qq)


def gen_analysis_window_noise(sigma_wn, sos_bp, sos_lp, rng, n_pad=4096):
    """
    Generate one full 2^19-sample amplitude noise trace via bandpass + lowpass
    filtered white Gaussian noise (same filter chain as process_gas_data.py).
    """
    total = n_pad + analysis_window_length
    white = rng.normal(0, sigma_wn, total)
    bp    = sosfilt(sos_bp, white)
    lp    = sosfilt(sos_lp, bp)
    return lp[n_pad:]


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
    idx_in_window : (n_searches,) absolute indices within the analysis window
    """
    ub           = analysis_window_length - search_window_length
    amp_search   = amp_lp[lb:ub]
    n_sw         = amp_search.size // search_window_length
    amp_reshaped = amp_search[:n_sw * search_window_length].reshape(n_sw, search_window_length)
    local_idx    = np.argmax(np.abs(amp_reshaped), axis=1)
    idx_in_win   = (local_idx + lb
                    + search_window_length * np.arange(n_sw, dtype=np.int32))
    amps         = amp_lp[idx_in_win]
    return amps, idx_in_win


def calc_chi2_all(amp_lp, idx_in_window, normalized_template_chi2, sigma_p_amp):
    """
    Vectorised chi-squared for all sub-windows using the 500-sample template,
    matching calc_chisquares() in process_gas_data.py (window_size=250).

    Returns np.inf for edge sub-windows where the 500-sample window would fall
    outside amp_lp.
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


# ── Main MC loop ───────────────────────────────────────────────────────────

def run_mc(n_windows, events_per_window, rng,
           sos_bp, sos_lp, sigma_wn, sigma_p_amp,
           ps_norm, peak_cal, normalized_template_chi2,
           qq_kev, drdqz,
           amp2kev_val,
           noise_thr_kev=200.0, chi2_thr=700.0, apply_dc=True):
    """
    Run MC simulation over n_windows full analysis windows.

    For each window:
      1. Generate bandlimited Gaussian noise.
      2. Inject Poisson(events_per_window) signal events (random positions, ±1 signs).
      3. Search: argmax |amp| in each 256-sample sub-window.
      4. Noise-level cut   (per window):   std(searchable region) < noise_thr_kev.
      5. Chi-squared cut   (per sub-window): chi2 < chi2_thr.
      6. Double-count rejection.

    Parameters
    ----------
    noise_thr_kev : pass np.inf to disable noise-level cut
    chi2_thr      : pass np.inf to disable chi-squared cut
    apply_dc      : set False to skip double-count rejection

    Returns
    -------
    q_meas_kev : np.ndarray — passing |amplitudes| in keV/c
    n_good     : int        — number of analysis windows that passed noise-level cut
    """
    q_meas_list = []
    n_good      = 0

    for _ in range(n_windows):
        amp_lp = gen_analysis_window_noise(sigma_wn, sos_bp, sos_lp, rng)

        # Signal injection
        n_events = rng.poisson(events_per_window)
        if n_events > 0:
            q_true = sample_spectrum_icdf(qq_kev, drdqz, n_events, rng)
            signs  = rng.integers(0, 2, size=n_events) * 2 - 1
            positions = rng.integers(
                lb, analysis_window_length - search_window_length, size=n_events
            )
            inject_signals(amp_lp, q_true * signs, positions, ps_norm, peak_cal, amp2kev_val)

        # Search
        amps, idx_win = search_and_recon(amp_lp)

        # Noise-level cut
        noise_level_kev = (
            np.std(amp_lp[lb : analysis_window_length - search_window_length]) * amp2kev_val
        )
        if noise_level_kev >= noise_thr_kev:
            continue
        n_good += 1

        # Chi-squared cut
        chi2    = calc_chi2_all(amp_lp, idx_win, normalized_template_chi2, sigma_p_amp)
        amps_2d = amps.reshape(1, -1)
        idx_2d  = idx_win.reshape(1, -1)
        if chi2_thr < np.inf:
            amps_2d[0, chi2 >= chi2_thr] = np.nan

        # Double-count rejection
        if apply_dc:
            amps_2d = throw_away_doublecounts(
                amps_2d, np.array([True]), idx_2d, amp2kev=amp2kev_val
            )

        passing = amps_2d[0]
        passing = np.abs(passing[~np.isnan(passing)]) * amp2kev_val
        q_meas_list.extend(passing)

    return np.array(q_meas_list), n_good


# ============================================================
# MAIN
# ============================================================

def main():
    global amp2kev   # allow config-block value to be passed into helper that uses it

    # ── Resolve gas mass ──────────────────────────────────────────────────
    gas_key = gas.lower()
    if gas_key not in GAS_MASSES_AMU:
        raise ValueError(
            f"Unknown gas '{gas}'. Choose from: {list(GAS_MASSES_AMU.keys())}"
        )
    mg_amu = GAS_MASSES_AMU[gas_key]

    # ── Effective cut settings ────────────────────────────────────────────
    noise_thr = noise_threshold_kev if apply_noise_cut else np.inf
    chi2_thr  = chi2_threshold      if apply_chi2_cut  else np.inf
    dc        = apply_dc_cut

    cuts_tag = '_'.join([
        'noise' if apply_noise_cut  else 'nonoise',
        'chi2'  if apply_chi2_cut   else 'nochi2',
        'dc'    if apply_dc_cut     else 'nodc',
    ])

    # ── Output tag ────────────────────────────────────────────────────────
    tag = output_tag or f'{sphere}_{gas_key}_{cuts_tag}'

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f'{tag}.npz')

    # ── Filters ───────────────────────────────────────────────────────────
    sos_bp = butter(lowpass_order, [bandpass_lb, bandpass_ub], 'bandpass', fs=fs, output='sos')
    sos_lp = butter(lowpass_order, bandpass_ub, 'lp', fs=fs, output='sos')

    # ── Pulse template ────────────────────────────────────────────────────
    tpl_path = pulse_template_path or _resolve_template_path(sphere)
    ps_norm, peak_cal, normalized_template_chi2, fwhm_samps = _load_pulse_template(
        tpl_path, search_window_length
    )

    # ── Noise calibration ─────────────────────────────────────────────────
    sigma_wn   = _calibrate_noise(sos_bp, sos_lp, sigma_noise_kev, amp2kev)
    sigma_p_amp = sigma_noise_kev / amp2kev

    # ── Theory spectrum at unit pressure (1e-8 mbar) ──────────────────────
    qq_kev = np.linspace(0.5, 2000, 2000)
    drdq_unit = calc_gas.dgamma_dp_tot_noneq(
        qq_kev, mg_amu, p_mbar=1e-8, alpha=alpha,
        T_gas=T_gas, T_sensor=T_sensor, sphere_radius=sphere_radius
    )
    _, drdqz_unit = calc_gas.get_drdqz(qq_kev, drdq_unit)
    total_rate_unit = np.trapz(drdqz_unit, qq_kev)   # Hz at 1e-8 mbar

    bc  = 0.5 * (bins[:-1] + bins[1:])
    dq  = float(bins[1] - bins[0])

    print(f'Sphere:         {sphere}')
    print(f'Gas:            {gas.upper()}  ({mg_amu} amu)')
    print(f'T_gas={T_gas} K   T_sensor={T_sensor} K   alpha={alpha}')
    print(f'Cuts:           noise={apply_noise_cut}  chi2={apply_chi2_cut}  dc={apply_dc_cut}')
    print(f'MC windows:     {n_analysis_windows} per pressure case')
    print(f'Rate @ 1e-8 mbar: {total_rate_unit:.3f} Hz')
    print()

    # ── Noise-only run ────────────────────────────────────────────────────
    print(f'Noise-only run ({n_analysis_windows} windows)…')
    rng_noise = np.random.default_rng(rng_seed)
    q_floor, n_floor_good = run_mc(
        n_analysis_windows, events_per_window=0, rng=rng_noise,
        sos_bp=sos_bp, sos_lp=sos_lp, sigma_wn=sigma_wn, sigma_p_amp=sigma_p_amp,
        ps_norm=ps_norm, peak_cal=peak_cal,
        normalized_template_chi2=normalized_template_chi2,
        qq_kev=qq_kev, drdqz=drdqz_unit,
        amp2kev_val=amp2kev,
        noise_thr_kev=noise_thr, chi2_thr=chi2_thr, apply_dc=dc,
    )
    hh_noise, _ = np.histogram(q_floor, bins)
    norm_noise   = np.sum(hh_noise) * T_search
    rate_noise   = hh_noise / norm_noise if norm_noise > 0 else np.zeros_like(hh_noise, float)
    print(f'  {n_floor_good}/{n_analysis_windows} good windows  |  '
          f'{len(q_floor):,} sub-windows  |  '
          f'noise floor median = {np.median(q_floor):.0f} keV/c\n')

    # ── Signal runs ───────────────────────────────────────────────────────
    hh_meas_all   = []
    rate_meas_all = []
    total_rates   = []
    n_good_all    = []

    for p_idx, p in enumerate(pressures_mbar):
        scale = p / 1e-8
        drdqz_p = drdqz_unit * scale
        total_rate_p = total_rate_unit * scale
        ev_per_win   = total_rate_p * T_analysis_window

        label = f'{p:.1e} mbar'
        print(f'Pressure {p_idx+1}/{len(pressures_mbar)}: {label}  '
              f'({ev_per_win:.3f} events/window)…')

        rng_sig = np.random.default_rng(rng_seed + 1 + p_idx)
        q_meas, n_good = run_mc(
            n_analysis_windows, ev_per_win, rng=rng_sig,
            sos_bp=sos_bp, sos_lp=sos_lp, sigma_wn=sigma_wn, sigma_p_amp=sigma_p_amp,
            ps_norm=ps_norm, peak_cal=peak_cal,
            normalized_template_chi2=normalized_template_chi2,
            qq_kev=qq_kev, drdqz=drdqz_p,
            amp2kev_val=amp2kev,
            noise_thr_kev=noise_thr, chi2_thr=chi2_thr, apply_dc=dc,
        )

        hh, _ = np.histogram(q_meas, bins)
        norm   = np.sum(hh) * T_search
        rate   = hh / norm if norm > 0 else np.zeros_like(hh, float)

        hh_meas_all.append(hh)
        rate_meas_all.append(rate)
        total_rates.append(total_rate_p)
        n_good_all.append(n_good)

        print(f'  {n_good}/{n_analysis_windows} good windows  |  {len(q_meas):,} passing sub-windows')

    # ── Save ──────────────────────────────────────────────────────────────
    np.savez(
        out_path,
        # Histogram axes
        bins            = bins,
        bc              = bc,
        dq              = dq,
        # Noise floor
        hh_noise        = hh_noise,
        rate_noise      = rate_noise,
        n_floor_good    = np.array(n_floor_good),
        # Per-pressure results
        pressures_mbar  = np.array(pressures_mbar),
        total_rates_hz  = np.array(total_rates),
        hh_meas         = np.array(hh_meas_all),    # (n_pressures, n_bins)
        rate_meas       = np.array(rate_meas_all),   # (n_pressures, n_bins)
        n_good          = np.array(n_good_all),
        # Theory spectrum (at unit pressure 1e-8 mbar; scale by p/1e-8 for other pressures)
        qq_kev          = qq_kev,
        drdqz_unit      = drdqz_unit,
        # Configuration metadata
        sphere          = np.array(sphere),
        gas             = np.array(gas.lower()),
        mg_amu          = np.array(mg_amu),
        amp2kev         = np.array(amp2kev),
        sphere_radius   = np.array(sphere_radius),
        bandpass_lb     = np.array(bandpass_lb),
        bandpass_ub     = np.array(bandpass_ub),
        sigma_noise_kev = np.array(sigma_noise_kev),
        T_gas           = np.array(T_gas),
        T_sensor        = np.array(T_sensor),
        alpha           = np.array(alpha),
        n_analysis_windows = np.array(n_analysis_windows),
        rng_seed        = np.array(rng_seed),
        apply_noise_cut = np.array(apply_noise_cut),
        apply_chi2_cut  = np.array(apply_chi2_cut),
        apply_dc_cut    = np.array(apply_dc_cut),
        noise_threshold_kev = np.array(noise_threshold_kev),
        chi2_threshold  = np.array(chi2_threshold),
    )
    print(f'\nSaved → {out_path}')


if __name__ == '__main__':
    main()
