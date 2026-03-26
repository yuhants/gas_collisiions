# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Physics analysis framework for studying gas collision recoils on levitated charged nanospheres (~50 nm radius, ~3e charge). The experiment detects momentum transfer from individual gas molecules (primarily xenon) colliding with a mechanically oscillating sphere, measured as impulse amplitudes in the range 250 keV–10 MeV.

## Running Scripts

Scripts are run directly from the repo root. Each script has configuration variables near the top that must be edited before running:

```bash
python process_impulse_calibration.py   # Process calibration HDF5 files
python process_gas_data.py              # Process gas run HDF5 files
python calc_gas_collision_spectrum.py   # Calculate theoretical collision spectra
python gas_likelihood_fit.py            # Run likelihood optimization
jupyter notebook                        # Launch notebooks for analysis
```

No test suite exists. No build step.

## Dependencies

```bash
pip install numpy scipy h5py matplotlib cycler
```

## Data Pipeline

```
Raw HDF5 (/Volumes/LaCie/gas_collisions/)
  → process_impulse_calibration.py → data_processed/pulse_calibration/*.hdf5
  → process_gas_data.py            → data_processed/*.hdf5
  → [notebooks: histogram binning] → data_processed/gas_recon/*.h5py
  → calc_gas_collision_spectrum.py → data_processed/gas_signal/*.npz
  → gas_likelihood_fit.py          → data_processed/likelihood_fit/*.npz
```

Raw data lives on an external drive (`/Volumes/LaCie/`). Processed data is committed under `data_processed/`.

## Architecture

**`analysis_utils.py`** — Core utility library imported by all scripts and notebooks:
- Signal filtering: `notch_filtered()`, `bandpass_filtered()`, `lowpass_filtered()`, `highpass_filtered()`
- Calibration: `get_c_mv()` (voltage-to-displacement), `get_effective_force_noise()`
- Pulse reconstruction: `get_pulse_idx()`, `recon_pulse()`, `get_unnormalized_amps()`
- PSD: `get_psd()` (Welch method), `get_area_driven_peak()`

**`calc_gas_collision_spectrum.py`** — Theoretical kinetic gas model:
- `dgamma_dp_tot_noneq()` — Non-equilibrium collision rate (main function used in fitting)
- `smear_drdqz_gauss()` — Convolves theory spectrum with Gaussian detector resolution
- `get_drdqz()` — Converts momentum transfer → detectable recoil energy

**`process_gas_data.py`** — Batch-processes raw xenon/background/krypton/SF6 runs:
- Two modes: sliding-window (`process_dataset`) and cal-mode (`process_dataset_cal_mode --cal`)
- Cal-mode triggers on cal pulses in gas runs, outputs `{dataset}_cal_processed.hdf5` with per-pulse arrays
- Fixed-parameter overrides: `fixed_gamma_damping`, `fixed_c_imp` bypass Voigt fit entirely
- `SCAN_C_IMP`: scans c_imp_scaling on pilot files; skipped when `fixed_c_imp` is set
- HDF5 attrs: `fixed_c_imp` OR `c_imp_scaling` (mutually exclusive), optionally `fixed_gamma_damping`
- chi2 convention: `sum(((waveform - amp*T_norm) / sigma_amp)^2)`, `sigma_amp = 60keV / amp2kev`, window = 500 samples (±250 around peak)

**`process_impulse_calibration.py`** — Processes known-force calibration shots at 2.5V–20V:
- Fixed-parameter overrides: `fixed_gamma_damping`, `fixed_c_imp` (same convention as process_gas_data.py)
- At 2.5V, also reconstructs noise-only windows (midpoints between cal pulses) and saves:
  - `noise_waveforms_2.5v` — waveform centred at geometric midpoint (no search)
  - `noise_waveforms_search_2.5v` — waveform centred at searched peak (consistent with signal pulses)
  - `noise_drive_area_2.5v`, `noise_f_res_2.5v`, `noise_noise_level_2.5v` — diagnostics for quality cuts
- All noise arrays are aligned (same entries); if either waveform slice is out-of-bounds, the whole noise index is skipped

**`recon_histograms.py`** — Builds amplitude histograms from sliding-window processed output:
- `get_summed_histogram()` returns `(bc, hh_all, hh_nocal, hh_cal)` — 4 values
- Quality cuts: `noise_threshold_kev=70`, `chi2_threshold=700`, `normalized_drive_power_threshold=4.5e-9`
- Double-count removal: same-peak straddling and peak+ring-down trough cases
- `flag_cal_pulses()`: timing window `[pulse_index+20, pulse_index+276)`, amp threshold 700 keV/c
- Output HDF5 has `{dataset}`, `{dataset}_nocal`, `{dataset}_cal` datasets per gas type group

**`gas_likelihood_fit.py`** — Likelihood optimization:
- `calc_nll()` — Negative log-likelihood comparing data histogram to theory + background model
- `minimize_nll()` — Nelder-Mead optimizer over (sigma_keV, log10_pressure)
- Detection efficiency modeled as error function: `func_eff()`

## Key Parameters and Conventions

### Sphere-specific configuration (edit in scripts per sphere)
```python
# sphere_20260215 (current active sphere, after introducing imprecision):
bandpass_lb, bandpass_ub = (35000, 80000)   # Hz
notch_freq = 137000                          # Hz
fixed_gamma_damping = 1 * 2 * np.pi         # rad/s
fixed_c_imp = 1.5e-22                        # raw units
amp2kev = 14460.84503586                     # keV/c per raw unit

# sphere_20260105:
bandpass_lb, bandpass_ub = (39000, 74000)
amp2kev = 6792.86

# sigma_amp for chi2:
sigma_amp = 60 / amp2kev                     # 60 keV/c noise assumption
```

### Dataset naming
- Datasets: `YYYYMMDD_p{N_elements}e_{pressure}mbar[_tag]` — e.g., `20260107_p8e_4e-8mbar`
- Spheres: `sphere_YYYYMMDD` (creation date)
- Processed files (sliding-window): `{dataset}/{file_prefix}{i}_processed.hdf5`
- Processed files (cal-mode): `{dataset}/{dataset}_cal_processed.hdf5`
- Calibration processed: `pulse_calibration/{sphere}/{dataset}_processed.hdf5`

### HDF5 structure — sliding-window output (`process_gas_data.py`)
```
data_processed/
  amplitude          (n_windows, n_searches)  float64
  idx_in_window      (n_windows, n_searches)  int32
  good_detection     (n_windows,)             bool
  noise_level_amp    (n_windows,)             float64
  f_res              (n_windows,)             float64
  driven_power       (n_windows,)             float64
  chisquare          (n_windows, n_searches)  float64
  cal_pulse_indices  (n_cal,)                 int32
  attrs: pressure_mbar, amp2kev, fixed_c_imp | c_imp_scaling, [fixed_gamma_damping]
```

### HDF5 structure — cal-mode output (`process_gas_data.py --cal`)
```
data_processed/
  amplitude          (N,)        float64
  waveform           (N, 3000)   float64
  noise_level_amp    (N,)        float64
  driven_power       (N,)        float64
  f_res              (N,)        float64
  idx_in_window      (N,)        int32
  file_index         (N,)        int32
  pulse_abs_index    (N,)        int32
  attrs: fixed_c_imp | c_imp_scaling, [fixed_gamma_damping]
```

### HDF5 structure — impulse calibration output (`process_impulse_calibration.py`)
```
data_processed/
  amplitudes_{v}v              (N,)       float64   # per voltage
  pulse_shapes_{v}v            (N, 3000)  float64
  noise_level_{v}v             (N,)       float64
  drive_area_{v}v              (N,)       float64
  f_res_{v}v                   (N,)       float64
  pulse_indices_in_win_{v}v    (N,)       int32
  z_signal_{v}v                (N, 3000)  float64
  # 2.5V only:
  amplitudes_noise_2.5v        (M,)       float64   # amplitude at midpoint (no search)
  amplitudes_noise_search_2.5v (M,)       float64   # searched peak amplitude
  noise_waveforms_2.5v         (M, 3000)  float64   # waveform at midpoint
  noise_waveforms_search_2.5v  (M, 3000)  float64   # waveform at searched peak
  noise_drive_area_2.5v        (M,)       float64
  noise_f_res_2.5v             (M,)       float64
  noise_noise_level_2.5v       (M,)       float64
  attrs: fixed_c_imp | c_imp_scaling, [fixed_gamma_damping]
```

### Analysis window sizes
```python
analysis_window_length = 2**19  # ~105 ms at 5 MHz sampling
search_window_length   = 2**8   # ~51 µs
fit_window_length      = 2**19  # prepulse PSD window
waveform_half_len      = 1500   # samples each side of peak (3000 total)
template_half_len      = 250    # chi2 window (500 total)
```

### Quality cuts (sphere_20260215)
These values are subject to change — `recon_histograms.py` has the authoritative current values.
```python
noise_threshold_kev              = 70      # keV/c (noise_level_amp * amp2kev)
normalized_drive_power_threshold = 4.5e-9  # drive_area * (f_res²-drive_freq²)² / (ref_freq²-drive_freq²)²
chi2_threshold                   = 600     # total chi2 (not per-dof)
```

## Active Experiments (Mar 2026)

- **sphere_20260215**: Primary active sphere; xenon/krypton/SF6/background runs at multiple pressures; using `fixed_gamma_damping = 1*2π rad/s`, `fixed_c_imp = 1.5e-22`, `amp2kev = 14460.85`
- **sphere_20260105**: Earlier sphere; xenon runs at 4e-8 to 2e-7 mbar; `amp2kev = 6792.86`

## Analysis Notebooks Location

Organized by sphere under `analysis_notebooks/sphere_YYYYMMDD/`:
- `analysis_notebooks/sphere_20260215_final_analysis/` — Active final analysis for sphere_20260215
  - `20260320_sphere_20260215_impulse_calibration.ipynb` — Calibration + noise characterisation; chi2 diagnostics; gas dataset cal-pulse quality
  - `20260319_sphere_20260215_cimp_scan_xe.ipynb` — c_imp scan results for xenon
- `*_calibration.ipynb` — Voltage-to-energy calibration
- `*_likelihood_fit.ipynb` — Main fit and results
- `*_recon.ipynb` — Pulse reconstruction and histogram generation
- `*_noise_analysis.ipynb` — Noise characterization
