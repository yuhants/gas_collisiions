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

**`process_gas_data.py`** — Batch-processes raw xenon/background runs:
- Applies notch filter (137 kHz drive tone removal), bandpass (sphere-specific, e.g. 39–74 kHz)
- Reconstructs impulse amplitudes from force time-series using susceptibility inversion
- Outputs per-window quality flags (`good_detection`), amplitudes, chi-squared, noise levels

**`process_impulse_calibration.py`** — Processes known-force calibration shots at multiple drive voltages (2.5V–20V), building a calibration HDF5 with pulse shapes, amplitudes, and resonant frequencies.

**`gas_likelihood_fit.py`** — Likelihood optimization:
- `calc_nll()` — Negative log-likelihood comparing data histogram to theory + background model
- `minimize_nll()` — Nelder-Mead optimizer over (sigma_keV, log10_pressure)
- Detection efficiency modeled as error function: `func_eff()`

## Key Parameters and Conventions

### Sphere-specific configuration (edit in scripts per sphere)
```python
bandpass_lb, bandpass_ub = (39000, 74000)  # Hz — sphere_20260105
notch_freq = 137000                          # Hz
sigma_p_amp = 60 / amp2kev_factor           # Amplitude noise (keV)
```

### Dataset naming
- Datasets: `YYYYMMDD_p{N_elements}e_{pressure}mbar[_tag]` — e.g., `20260107_p8e_4e-8mbar`
- Spheres: `sphere_YYYYMMDD` (creation date)
- Processed files: `{dataset}_processed.hdf5`

### HDF5 structure (processed output)
```
/amplitude          (n_windows, n_searches)  float64
/idx_in_window      (n_windows, n_searches)  int16
/good_detection     (n_windows,)             bool
/noise_level_amp    (n_windows,)             float64
/f_res              (n_windows,)             float64
/driven_power       (n_windows,)             float64
/chisquare          (n_windows, n_searches)  float64
attrs: pressure_mbar, timestamp
```

### Analysis window sizes
```python
analysis_window_length = 2**19  # ~105 ms at 5 MHz sampling
search_window_length   = 2**8   # ~51 µs
prepulse_window_length = 50000  # 10 ms (for resonant frequency extraction)
```

## Active Experiments (Feb 2026)

- **sphere_20260105**: Primary analysis target; xenon runs at 4e-8 to 2e-7 mbar
- **sphere_20260215**: Latest sphere; pulse calibration data recently acquired

## Analysis Notebooks Location

Organized by sphere under `analysis_notebooks/sphere_YYYYMMDD/`:
- `*_calibration.ipynb` — Voltage-to-energy calibration
- `*_likelihood_fit.ipynb` — Main fit and results
- `*_recon.ipynb` — Pulse reconstruction and histogram generation
- `*_noise_analysis.ipynb` — Noise characterization
