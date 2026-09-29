% ========================================================================
% main.m -- Full DyPLS-IL-MPC simulation pipeline (Octave-compatible).
%
% Steps:
%   1. Initialize parameters              init_params
%   2. Generate excitation data           gen_excitation
%   3. Build DyPLS-ARX model              build_dypls
%   4. Estimate uncertainty bounds        estimate_uncertainty
%   5. Compute Tube-MPC parameters        compute_tube
%   6. Run ILC-MPC online simulation      run_ilmpc
%   7. Plot results (optional)            plot_results
%   8. Export all figures to EPS/PDF/PNG  export_all_figures
%
% Notes:
%   * Script-internal local functions are avoided for compatibility with
%     older Octave releases; all try/catch blocks are inlined.
%   * Random seed is set via Params.rng_seed for reproducibility.
%   * Plot switches: Params.plot_excitation, Params.plot_model,
%                    Params.plot_results.
%   * Figure export is controlled by Params.export_figures (default true)
%     and Params.figures_dir (default './figures').
% ========================================================================
clear; clc; close all;

if exist('OCTAVE_VERSION', 'builtin')
    pkg load control optim;
end

fprintf('=== DyPLS-IL-MPC Simulation ===\n');

% ========================================================================
% Step 1. Initialize parameters
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

% -- Default export-related switches (safe if init_params didn't set them) --
if ~isfield(Params, 'export_figures') || isempty(Params.export_figures)
    Params.export_figures = true;
end

if ~isfield(Params, 'figures_dir') || isempty(Params.figures_dir)
    Params.figures_dir = 'figures';
end

if isfield(Params, 'rng_seed') && ~isempty(Params.rng_seed)

    if exist('rng', 'builtin')
        rng(Params.rng_seed);
    else
        rand('seed', Params.rng_seed);
        randn('seed', Params.rng_seed);
    end

    fprintf('main: random seed set to %d.\n', Params.rng_seed);
end

% ========================================================================
% Step 2. Generate excitation data
% ========================================================================
try
    Identification_data = gen_excitation(Params);
catch ME
    fprintf(2, '[gen_excitation] failed: %s\n', ME.message);

    for k = 1:numel(ME.stack)
        fprintf(2, '  File: %s, line: %d, function: %s\n', ...
            ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
    end

    rethrow(ME);
end

fprintf('main: excitation data generated.\n');

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

fprintf('Y_mean = [%s]\n', num2str(Identification_data.Y_mean', '%.4f '));
fprintf('Y0 mean = [%s]\n', num2str(mean(Identification_data.Y0, 1), '%.4e '));
fprintf('Model.R_y rel err = %.4e\n', ...
    norm(Model.R_y, 'fro') / max(norm(Identification_data.Y0, 'fro'), eps));

T_check = Identification_data.X0 * Model.R_x;
rel_err = norm(T_check - Model.T, 'fro') / norm(Model.T, 'fro');
fprintf('T reconstruction error: %.4e\n', rel_err);
fprintf('max |R_x|     = %.4e\n', max(abs(Model.R_x(:))));
fprintf('max |pinv(R_x'')| = %.4e\n', max(abs(pinv(Model.R_x')(:))));

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
% Step 8. Export all figures (optional)
% ------------------------------------------------------------------------
% Exports the figures created in Step 7 to vector EPS / PDF and 300-dpi PNG.
% Controlled by Params.export_figures (logical) and Params.figures_dir
% (string path). Figures are matched by their Name prefix, so any figure
% that was not created (e.g. Model not supplied) is silently skipped.
% ========================================================================
if Params.plot_results && Params.export_figures

    try
        export_all_figures(Params, Params.figures_dir);
    catch ME
        fprintf(2, '[export_all_figures] failed: %s\n', ME.message);

        for k = 1:numel(ME.stack)
            fprintf(2, '  File: %s, line: %d, function: %s\n', ...
                ME.stack(k).file, ME.stack(k).line, ME.stack(k).name);
        end

        % Export failure should not abort the whole run; just warn.
        warning('main: figure export failed; continuing.');
    end

    fprintf('main: figure export completed (dir = "%s").\n', ...
        Params.figures_dir);
elseif ~Params.plot_results
    fprintf('main: figure export skipped (plot_results = false).\n');
else
    fprintf('main: figure export skipped (export_figures = false).\n');
end

fprintf('=== Simulation completed ===\n');
