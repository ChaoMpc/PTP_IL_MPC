% ========================================================================
% main.m -- Full DyPLS-IL-MPC simulation pipeline (Octave-compatible).
%
% Plant : Wood-Berry binary distillation column (2x2)
%
% Important: Wood-Berry G(s) is an INCREMENTAL model
% --------------------------------------------------
%   Delta y(s) = G(s) * Delta u(s),  Delta u = u - u_ss, Delta y = y - y_ss
%
%   All internal computation is done in PHYSICAL DEVIATION form:
%       v(k) = u(k) - u_ss   [lb/min]
%       y(k) = y(k) - y_ss   [wt%]
%   Constraint and reference-point conversions from absolute physical
%   values to physical deviations are performed ONCE in init_params.
%   Absolute values are reconstructed in plot_results by adding
%   Params.y_ss / Params.u_ss for display.
%
% Steps
% -----
%   1. Initialize parameters                       init_params
%      (single conversion point: absolute -> physical deviation)
%   2. Generate excitation data (deviation form)   gen_prbs_excitation
%   3. Build DyPLS-ARX model                       build_dypls
%   4. Estimate uncertainty bounds                 estimate_uncertainty
%   5. Compute Tube-MPC parameters                 compute_tube
%   6. Run ILC-MPC online simulation               run_ilmpc
%   7. Plot results                                plot_results
%   8. Export figures to EPS / PDF / PNG           export_all_figures
%
% Notes
% -----
%   * Random seed is set via Params.rng_seed for reproducibility.
%   * Plot switches: Params.plot_excitation, Params.plot_model,
%                    Params.plot_results.
% ========================================================================
clear; clc; close all;

if exist('OCTAVE_VERSION', 'builtin')
    pkg load control optim;
end

fprintf('=== DyPLS-IL-MPC Simulation (Wood-Berry, physical-deviation form) ===\n');

% ========================================================================
% Step 1. Initialize parameters
% ------------------------------------------------------------------------
% All conversions from absolute physical values to physical deviations
% (constraints, reference points) are performed inside init_params.
% ========================================================================
try
    Params = init_params();
catch ME
    fprintf(2, '[init_params] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

if isfield(Params, 'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng', 'builtin')
        rng(Params.rng_seed);
    else
        rand('seed',  Params.rng_seed);
        randn('seed', Params.rng_seed);
    end
    fprintf('main: random seed set to %d.\n', Params.rng_seed);
end

% ========================================================================
% Step 2. Generate excitation data (deviation form)
% ------------------------------------------------------------------------
% This step ONLY generates identification data (X_dev/Y_dev and their
% standardized versions X0/Y0). Constraint conversion and reference-point
% conversion are already done in init_params and are NOT touched here.
% ========================================================================
try
    [Identification_data, Params] = gen_prbs_excitation(Params);
catch ME
    fprintf(2, '[gen_prbs_excitation] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

fprintf('main: excitation data generated (deviation form).\n');

% ========================================================================
% Step 3. Build DyPLS-ARX model
% ========================================================================
try
    Model = build_dypls(Identification_data, Params);
catch ME
    fprintf(2, '[build_dypls] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

fprintf('main: DyPLS-ARX model built.\n');

% ========================================================================
% Step 4. Estimate uncertainty bounds
% ========================================================================
try
    UncSets = estimate_uncertainty(Model, Identification_data, Params);
catch ME
    fprintf(2, '[estimate_uncertainty] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

fprintf('main: uncertainty bounds estimated.\n');

% ========================================================================
% Step 5. Compute Tube-MPC parameters
% ------------------------------------------------------------------------
% compute_tube reads Params.h_v_phys / h_y_phys (already physical
% deviations from init_params) and outputs Tube.h_v_tight / h_y_tight in
% the same physical-deviation space.
% ========================================================================
try
    Tube = compute_tube(Identification_data, Model, UncSets, Params);
catch ME
    fprintf(2, '[compute_tube] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

fprintf('main: Tube-MPC parameters computed.\n');
fprintf('  Tube closed-loop spectral radius = %.6f\n', Tube.spectral_radius);

if Tube.spectral_radius >= 1
    warning('main: Tube closed-loop spectral radius >= 1; gain is not stabilizing.');
end

% ========================================================================
% Step 6. Run ILC-MPC
% ------------------------------------------------------------------------
% run_ilmpc operates entirely on physical deviations. Constraints come
% from Tube.h_v_tight / h_y_tight; references come from Params.Ref_points.
% ========================================================================
try
    Data = run_ilmpc(Identification_data, Model, Tube, UncSets, Params);
catch ME
    fprintf(2, '[run_ilmpc] failed: %s\n', ME.message);
    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end
    rethrow(ME);
end

fprintf('main: ILC-MPC simulation completed.\n');

% ========================================================================
% Step 7. Plot results (optional)
% ------------------------------------------------------------------------
% plot_results adds u_ss / y_ss back to the physical-deviation data for
% display, and uses the absolute physical bounds/references for plotting.
% ========================================================================
if isfield(Params, 'plot_results') && Params.plot_results
    try
        plot_results(Data, Params, Model, Tube);
    catch ME
        fprintf(2, '[plot_results] failed: %s\n', ME.message);
        for k = 1:numel(ME.stack)
            fprintf(2, '  File: %s, line: %d, function: %s\n', ...
                ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
        end
        rethrow(ME);
    end
    fprintf('main: plotting completed.\n');
else
    fprintf('main: plotting skipped (Params.plot_results = false).\n');
end

% ========================================================================
% Step 8. Export all figures to EPS / PDF / PNG
% ========================================================================
if isfield(Params, 'plot_results') && Params.plot_results
    try
        export_all_figures(Params, 'figures');
    catch ME
        fprintf(2, '[export_all_figures] failed: %s\n', ME.message);
        for k = 1:numel(ME.stack)
            fprintf(2, '  File: %s, line: %d, function: %s\n', ...
                ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
        end
    end
    fprintf('main: figures exported to figures/\n');
else
    fprintf('main: figure export skipped (plot_results = false).\n');
end

fprintf('=== Simulation completed ===\n');
