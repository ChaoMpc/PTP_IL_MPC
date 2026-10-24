% =========================================================================
% run_open_loop_test.m -- Convenience wrapper for cryst_open_loop_test.
%
% Run this script BEFORE main.m to understand the crystallization plant's
% open-loop behavior, then use the printed recommendations to adjust
% Params.y_phys_min/max, Params.Ref_points and Params.prbs_amp in init_params.m.
% =========================================================================

clear; clc; close all;
if exist('OCTAVE_VERSION','builtin'), pkg load control optim; end

Params = init_params();

% Optional tuning:
%   opt.step_scale : scalar multiplier on Params.prbs_amp (default 1.0)
%   opt.k_step     : index at which the step is applied     (default 10)
%   opt.make_plots : true/false                             (default true)
opt = struct('step_scale', 1.0, 'k_step', 10, ...
             'make_plots', true, 'verbose', true);

Result = cryst_open_loop_test(Params, opt);

% Persist
if ~exist('figures','dir'), mkdir('figures'); end
save('-mat', fullfile('figures','open_loop_result.mat'), 'Result', 'Params');

fprintf('\nOpen-loop test completed.\n');
fprintf('Next steps:\n');
fprintf('  1. Inspect figures in the current figure window.\n');
fprintf('  2. Update Params.y_phys_min/max and Params.Ref_points in init_params.m\n');
fprintf('     based on the recommended ranges printed above.\n');
fprintf('  3. Run main.m to execute the closed-loop IL-MPC simulation.\n');
