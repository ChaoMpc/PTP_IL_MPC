# DyPLS-IL-MPC Simulation Pipeline

This repository contains the Octave source code for the simulation pipeline of the **PTP-IL-MPC** (Point-to-point - Iterative Learning Model Predictive Control) method. 

The code generates the simulation results, figures, and tables presented in the paper. It covers system excitation, DyPLS-ARX model construction, uncertainty bound estimation, Tube-MPC parameter computation, and ILC-MPC online simulation.

## Requirements

- ** GNU Octave** (Tested with Octave 6.0+). If using Octave, the `control` and `optim` packages are required.
- No additional MATLAB toolboxes are strictly required beyond the standard installation, but the `control` and `optim` packages are loaded if Octave is detected.

## File Structure

Ensure all the following files are in the same directory before running the main script:

- `main.m` - Main script to execute the full simulation pipeline.
- `init_params.m` - Initializes all simulation parameters.
- `gen_excitation.m` - Generates excitation data for system identification.
- `build_dypls.m` - Builds the DyPLS-ARX model.
- `estimate_uncertainty.m` - Estimates uncertainty bounds.
- `compute_tube.m` - Computes Tube-MPC parameters.
- `run_ilmpc.m` - Runs the ILC-MPC online simulation.
- `plot_results.m` - Plots the simulation results (optional).
- `export_all_figures.m` - Exports all figures to EPS/PDF/PNG formats (optional).

## How to Run

1. Open GNU Octave.
2. Navigate to the directory containing the source code.
3. Run the main script by typing the following command in the command window: main