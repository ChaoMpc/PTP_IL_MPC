================================================================================
  Example 2: Crystallization Batch Process
================================================================================

--------------------------------------------------------------------------------
1. OVERVIEW
--------------------------------------------------------------------------------
This toolbox implements a complete workflow for data-driven modeling and
iterative-learning model predictive control (IL-MPC) of a batch crystallization
process. It covers three layers:

  (a) Plant simulator          : cryst_plant.m
  (b) Offline system ID        : DyPLS (PLS latent variables) + ARX per latent
                                 variable, uncertainty quantification, and a
                                 robust tube for constraint tightening.
  (c) Online batch control     : Tube-based IL-MPC (proposed method) and an
                                 ILC baseline (Xue method) for comparison.

The intended usage is a two-step pipeline:

    Step 1 (offline) :  run_identification      -> produces identified_model.mat
    Step 2 (online)  :  main_online             -> runs closed-loop comparison

--------------------------------------------------------------------------------
2. ENVIRONMENT REQUIREMENTS
--------------------------------------------------------------------------------
  - MATLAB (R2018b or newer recommended), or GNU Octave (with "control" and
    "optim" packages). All files contain OCTAVE_VERSION checks and provide an
    Octave-compatible QP path (qp) in addition to quadprog.
  - No external toolboxes are required beyond MATLAB's Optimization Toolbox
    (quadprog) and Control System Toolbox (dlqr).

--------------------------------------------------------------------------------
3. QUICK START  (recommended workflow)
--------------------------------------------------------------------------------
  STEP 1 — System Identification (offline)
  ----------------------------------------
    >> run_identification

    This script will:
      * Load default parameters from init_params.m.
      * Generate PRBS excitation data for N_train_batches batches
        (gen_cryst_excitation.m).
      * Generate an independent validation set
        (gen_cryst_validation_data.m).
      * Build the DyPLS-ARX model (build_dypls.m) with A_lv latent variables
        and ARX order n_arx.
      * Report free-run NRMSE/VAF on the control phase (60-120 min) for the
        validation set (free_run_validation.m).
      * Estimate uncertainty bounds eta_bound / r_bound from the ARX
        innovations and PLS residuals (estimate_uncertainty.m).
      * Compute the robust tube (feedback gain K, invariant set, input/output
        tightenings delta_v / delta_y) via compute_tube.m.
      * Save everything into identified_model.mat
        (Ident_data, Val_data, Model, UncSets, Tube, CV_Data, Params).

  STEP 2 — Closed-loop Control (online)
  --------------------------------------
    >> main_online

    This script will:
      * Load identified_model.mat (must exist — run Step 1 first).
      * Compute the optimal reachable reference points on the tightened
        feasible set (compute_optimal_reachable.m) and print a comparison
        against Params.y_ref_phys.
      * Run the PROPOSED tube-based IL-MPC  (run_ilmpc.m).
      * Run the XUE baseline (pure ILC, no intra-batch feedback)
        (run_xue_ilmpc.m).
      * Evaluate both methods with compute_metrics.m (constraint violations,
        actuator saturation, final errors, QP timing).
      * Produce 7 comparison figures (plot_results.m) and export them to the
        "figures/" directory (export_all_figures.m).
      * Save results to simulation_results.mat and comparison_table.csv.

  ALL-IN-ONE alternative:
    >> main
    (Runs identification and control in a single script; slower to iterate but
     convenient if you want a one-shot reproduction of all results.)

--------------------------------------------------------------------------------
4. FILE-BY-FILE DESCRIPTION
--------------------------------------------------------------------------------
4.1 Entry scripts
-----------------
  run_identification.m   Offline identification pipeline (Step 1).
  main_online.m          Online control pipeline (Step 2); requires the .mat
                         produced by run_identification.
  main.m                 Combined identification + control in one script.
  run_tuning.m           Grid search over (A_lv, n_arx) using validation data
                         (tune_hyperparameters.m). Prints a copy-paste summary
                         of the best hyperparameters.
  run_open_loop_test.m   Convenience wrapper for cryst_open_loop_test.
  cryst_open_loop_analysis.m
                         Random-step open-loop exploration of all 6 input
                         channels; prints diagnostic ranges for constraint
                         tuning.
  diagnose_prbs.m        Visualizes PRBS excitation effect on the physical
                         plant (used to check input sensitivity before
                         identification).
  run_verify_profiles.m  Generates and validates baseline reference trajectories
                         (constant / cooling / aggressive cooling).
  test_xue.m             Standalone sanity test for the Xue-method state
                         estimator construction.

4.2 Plant & reference model
---------------------------
  cryst_plant.m          Nonlinear crystallization simulator: population
                         balance (mu0..mu3), energy balance (T, Tj) and
                         concentration; returns 6 sensor measurements.
  init_params.m          Central parameter file. Contains process constants,
                         time/sampling, PRBS settings, physical input/output
                         bounds, baseline trajectories (u_base, y_base),
                         tracking targets (Kp, Ref_points, track_idx),
                         LQR/MPC weights, and identification settings
                         (A_lv, n_arx, arx_lambda, ctrl_start).
                         ==> Modify this file to change any simulation or
                             control setting.

4.3 Identification utilities
----------------------------
  gen_cryst_excitation.m   Generates PRBS-based identification data (deviation
                           from baseline, standardized).
  gen_cryst_validation_data.m
                           Generates an independent validation set with a
                           different seed.
  build_dypls.m            Core identification routine:
                             - NIPALS PLS (static, A_lv components)
                             - ARX regression per latent variable using only
                               data from ctrl_start onward (with ridge
                               regularization)
                             - Builds the augmented state-space form
                               (barA, barB, barC, barE).
  estimate_uncertainty.m   Quantile-based bounds on ARX innovations and PLS
                           reconstruction residuals (control phase only).
  free_run_validation.m    Multi-step (free-run) and 1-step-ahead simulation
                           metrics NRMSE / VAF per output.
  report_pls_diagnostics.m Printed report: per-LV explained variance,
                           residual rank, conditioning.
  verify_pls_dimensionality.m
                           Quantitative evidence of PLS dimensionality
                           reduction (compression ratio, cond improvement,
                           QP dimension reduction).
  tune_hyperparameters.m   Grid-search hyperparameter selection.

4.4 Control utilities
---------------------
  run_ilmpc.m              PROPOSED tube-based IL-MPC. Per batch:
                             - Delta_t formulation of the QP (learning on the
                               latent trajectory)
                             - Tightened input/output constraints via delta_v /
                               delta_y
                             - Feedback compensation using the tube gain K
                             - Physical plant simulation with sensor noise.
                           Also injects an unmodeled disturbance at batches
                           15-16 (k = 80..85) for robustness comparison.
  run_xue_ilmpc.m          XUE baseline: ILC with an intra-batch reference
                           update, no tube feedback. Same disturbance schedule
                           as run_ilmpc for a fair comparison.
  compute_optimal_reachable.m
                           Solves a QP to find the best reachable tracking
                           point under tightened constraints; reports whether
                           the user reference is feasible.
  build_lifting_matrices.m Builds Phi_x, Phi_y, Phi_v and constant terms for
                           the finite-horizon prediction over N_ctrl steps.
  Initial_Feasible.m       Solves a minimum-effort QP to obtain a feasible
                           initial latent trajectory (used as t_prev for the
                           first IL-MPC batch).
  compute_metrics.m        Aggregates constraint-violation counts, max
                           violation magnitudes, saturation counts, and
                           uncertainty-set coverage.
  plot_results.m           Generates the 7 comparison figures.
  export_all_figures.m     Exports all open figures to EPS/PDF/PNG under
                           "figures/".

4.5 Verification helpers
------------------------
  verify_reference_trajectory.m
                           Builds and diagnoses a baseline input/output
                           trajectory for a given mode ('constant',
                           'cooling', 'cooling_aggr').
  cryst_open_loop_test.m   Full open-loop step-response test used to guide
                           constraint selection (baseline vs. step response).

--------------------------------------------------------------------------------
5. OUTPUT FILES
--------------------------------------------------------------------------------
  identified_model.mat     Produced by run_identification.m. Contains:
                             Ident_data, Val_data  — training / validation data
                             Model                 — DyPLS-ARX + state space
                             UncSets               — eta/r bounds
                             Tube                  — K, P, delta_v, delta_y
                             CV_Data               — free-run metrics
                             Params                — full parameter snapshot
  simulation_results.mat   Produced by main_online.m / main.m. Contains the
                           Data_prop, Data_xue, Metrics_prop, Metrics_xue
                           structures for post-processing.
  comparison_table.csv     Side-by-side metrics (violations, saturation,
                           final errors, QP times) for both methods.
  figures/                 Directory with all exported figures.

--------------------------------------------------------------------------------
6. TYPICAL WORKFLOW TIPS
--------------------------------------------------------------------------------
  * To change the horizon or tracking targets, edit init_params.m
    (Params.Kp, Params.Ref_points, Params.track_idx).
  * To tighten or relax physical constraints, edit Params.u_phys_min/max and
    Params.y_phys_min/max. Run cryst_open_loop_analysis.m first to see
    realistic ranges.
  * To change identification complexity, edit Params.A_lv and Params.n_arx;
    to search systematically, run run_tuning.m.
  * The control phase starts at Params.ctrl_start (default 60 min);
    identification uses only data at k >= ctrl_start + n_arx, i.e. only the
    portion of each batch where the PRBS excitation is active.
  * The disturbance schedule (batches 15-16, k=80..85) is hard-coded in both
    run_ilmpc.m and run_xue_ilmpc.m. Remove or modify if you do not want
    robustness tests.

--------------------------------------------------------------------------------
7. MINIMUM WORKING EXAMPLE
--------------------------------------------------------------------------------
    >> Params = init_params();          % load defaults
    >> run_identification;              % Step 1 (creates identified_model.mat)
    >> main_online;                     % Step 2 (runs IL-MPC vs Xue)

================================================================================
  End of README
================================================================================
