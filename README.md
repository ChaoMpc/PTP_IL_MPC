# PTP_IL_MPC
================================================================================

This is an example in our article.

MATLAB / GNU Octave implementation.

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------

This repository contains the complete, reproducible code. Example 1: Wood_Berry Columns

--------------------------------------------------------------------------------
2. HOW TO RUN
--------------------------------------------------------------------------------

Simply run:

    >> main

That is all. The script performs the full pipeline end-to-end:

    1. Parameter initialization            (init_params)
    2. PRBS identification data            (gen_prbs_excitation)
    3. Independent validation data         (gen_validation_data)
    4. DyPLS-ARX model identification      (build_dypls)
    5. Free-run validation                 (free_run_validation)
    6. Uncertainty set estimation          (estimate_uncertainty)
    7. Tube design and constraint tightening (compute_tube)
    8. Direct verification of unreachable reference (verify_unreachability)
    9. Closed-loop IL-MPC simulation       (run_ilmpc)
   10. Closed-loop metrics                 (compute_metrics)
   11. Initial-state perturbation test     (run_ilmpc, second pass)
   12. Plotting                            (plot_results)
   13. Figure export (EPS/PDF/PNG)         (export_all_figures)

Typical runtime: 30-90 seconds on a modern laptop.

All generated figures and diagnostic files are written to ./figures/.

Optional: hyperparameter tuning (A_lv and n_arx) can be run separately:

    >> run_tuning

This grid-searches A_lv x n_arx on the independent validation set using
free-run NRMSE and prints the best combination. The chosen values must be
copied manually into init_params.m.

--------------------------------------------------------------------------------
3. FUNCTION REFERENCE
--------------------------------------------------------------------------------

--- Entry points ---------------------------------------------------------------

main.m
    Top-level script. Runs the full closed-loop simulation and produces
    all figures, tables and diagnostic reports.

run_tuning.m
    Optional standalone script. Grid-searches A_lv and n_arx using the
    independent validation set and prints the best values.

--- Parameter and data generation ---------------------------------------------

init_params.m
    Defines all parameters: Wood-Berry transfer functions, sampling time,
    constraints, reference points, MPC/LQR weights, PRBS settings, noise
    variance, uncertainty bounds and test switches. All constraint and
    reference quantities are prepared in the physical deviation space.

gen_prbs_excitation.m
    Generates multi-channel PRBS excitation data for the Wood-Berry
    column. Returns the identification dataset (raw deviations X, Y, the
    standardized versions X0, Y0, and normalization constants). Also
    returns the updated Params struct.

gen_validation_data.m
    Generates independent validation data using a different random seed
    (Params.rng_seed_val). Not used for any model-parameter selection.

--- Modeling ------------------------------------------------------------------

build_dypls.m
    Core identification routine. Runs NIPALS to extract latent scores
    (T, U), fits an ARX model per latent channel via ridge-regularized
    least squares, and assembles the block-diagonal latent state-space
    model (barA, barB, barC, barE) plus the augmented incremental model
    (A_aug, B_aug, C_aug1, C_aug2). Also returns the PLS loadings
    (W, P, Q, R_x, R_y) and ARX parameters (Theta).

report_pls_diagnostics.m
    Prints PLS-layer diagnostics: per-component variance explained,
    truncation residual norm, condition numbers in the original and
    latent spaces, rank and null-space dimension of R_x. Called from
    build_dypls when Params.plot_model = true.

estimate_uncertainty.m
    Quantifies the two independent model-uncertainty sources:
        eta (ARX innovation, latent standardized deviation domain)
        r   (PLS output reconstruction residual)
    Each is bounded by safety_factor * column-wise empirical quantile at
    q_level. Returns eta_bound and r_bound, which drive the tube design.

--- Control design ------------------------------------------------------------

compute_tube.m
    Builds the finite-horizon tube. For each latent channel, solves an
    LQR, verifies Schur stability of the closed-loop, and accumulates
    the RPI ellipsoid P over P_horizon steps. Converts P into physical
    deviation tightening margins delta_v and delta_y, then produces the
    tightened bounds h_v_tight and h_y_tight. Also reports completeness
    of the uncertainty parameters for reproducibility.

build_lifting_matrices.m
    Builds the full-horizon lifting matrices Phi_x, Phi_y, Phi_v and the
    constant offsets x0_contrib, y0_const, v0_const. These are used by
    run_ilmpc, Initial_Feasible and verify_unreachability to express the
    predicted trajectory as a linear function of the latent input vector.

Initial_Feasible.m
    Solves a pure QP to generate a feasible initial latent input
    trajectory satisfying all tightened constraints. Guarantees that the
    first IL-MPC optimization is feasible.

--- Closed-loop control -------------------------------------------------------

run_ilmpc.m
    Executes the ILC-MPC closed loop:
        * Batch-to-batch: full-horizon QP solved once per batch.
        * Intra-batch: only tube feedback, no re-optimization.
    Also injects the initial-state perturbation (if enabled) and
    records empirical coverage of eta and r flags.

--- Validation and diagnostics -------------------------------------------------

free_run_validation.m
    Free-run simulation of the identified DyPLS-ARX model. Seeds only
    the first n_arx latent scores from the true data and generates all
    later values by ARX recursion. Returns NRMSE and VAF for each output.
    Supports both global and windowed evaluation.

verify_unreachability.m
    Solves the constrained optimal steady-state problem (Problem 55 in
    the paper) to find the best reachable setpoint y*_s. Used to justify
    that closed-loop trajectories converge to the physically reachable
    limit rather than the (unreachable) reference.

compute_metrics.m
    Aggregates closed-loop metrics: input/output constraint violations,
    empirical coverage of eta and r, and QP solve times. Prints a
    summary to the console.

tune_hyperparameters.m
    Called by run_tuning.m. Grid-searches A_lv and n_arx, evaluates
    free-run NRMSE on the validation set, and returns the best pair.

--- Visualization -------------------------------------------------------------

plot_results.m
    Produces Figures 1-8:
        Fig 1a/1b  Output tracking (y1, y2) with constraints and y*_s
        Fig 2a/2b  Control inputs (v1, v2)
        Fig 3      ILC convergence
        Fig 4      Free-run validation
        Fig 8a/8b  Initial-state perturbation test (y1, y2)
    Also prints the tracking-error table and saves it as .mat.

export_all_figures.m
    Exports all open figures to EPS, PDF and PNG at paper-compatible
    geometry. Files are written to ./figures/.

--------------------------------------------------------------------------------
4. OUTPUTS
--------------------------------------------------------------------------------

After running main.m, the ./figures/ directory contains:

    fig1a_tracking_y1.(eps|pdf|png)
    fig1b_tracking_y2.(eps|pdf|png)
    fig2a_inputs_v1.(eps|pdf|png)
    fig2b_inputs_v2.(eps|pdf|png)
    fig3_ilc.(eps|pdf|png)
    fig4_free_run.(eps|pdf|png)
    fig8a_init_y1.(eps|pdf|png)
    fig8b_init_y2.(eps|pdf|png)

    uncertainty_params.txt
        eta_bound, r_bound, W_eta, W_r, eig(P), delta_v, delta_y

    validation_table.csv
        Scenario vs (input violations, output violations, eta coverage,
        r coverage, mean QP time)

    tracking_error_table.mat
        Per-batch tracking errors at each reference point

--------------------------------------------------------------------------------
5. REQUIREMENTS
--------------------------------------------------------------------------------

MATLAB:
    R2018b or later
    Control System Toolbox
    Optimization Toolbox (provides qp)

GNU Octave:
    6.0 or later
    control and optim packages:
        >> pkg install -forge control optim
        >> pkg load control optim

main.m and plot_results.m detect Octave automatically and load the
required packages.

--------------------------------------------------------------------------------
6. NOTES
--------------------------------------------------------------------------------

Unreachable reference
    The Wood-Berry case has a physically reachable reference point.
    verify_unreachability.m computes the optimal reachable setpoint y*_s
    (shown as blue hollow circles in Fig. 1). Closed-loop trajectories
    converging to y*_s.

QP infeasibility
    If qp returns a non-zero status, check that h_v_tight and h_y_tight
    are strictly positive. compute_tube.m prints all tightening margins
    and reports whether the tightened set is non-empty. Reduce
    safety_factor if the tightening is too aggressive.

Coverage vs q_level
    eta_bound and r_bound are derived from empirical quantiles on the
    identification set. Slight deviation of the coverage on the
    independent validation set is normal.

Changing the plant
    Replace G and Gd in init_params.m. Also update n_in, n_out, u_ss,
    y_ss, u_phys_min, u_phys_max, y_phys_min, y_phys_max and all
    dimension-dependent fields.
    
================================================================================
