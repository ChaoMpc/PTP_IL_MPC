function Result = cryst_open_loop_test(Params, options)
% =========================================================================
% cryst_open_loop_test -- Open-loop step-response exploration for the
% crystallization plant.
%
% Purpose
% -------
% Identify each input channel's qualitative effect on each output. Used to
% guide the selection of:
%   * Params.Ref_points (which output to track, at what magnitude)
%   * Params.y_phys_min/max and Params.u_phys_min/max (constraint bounds)
%   * Params.prbs_amp (excitation amplitude for identification)
%
% Why "delta against a baseline"?
% --------------------------------
% The crystallization process starts supersaturated (C=0.0256 > C*~0.015),
% so even at constant u_ss the concentration drifts downward as crystallization
% proceeds. Simply comparing "final y - initial y" would conflate the input
% effect with this natural evolution. We therefore run two batches:
%   * Baseline: all inputs = u_ss, record Y_base(k)
%   * Step:     channel i = u_ss(i) + delta from k_step+1 onward
% and compute dY(k) = Y_step(k) - Y_base(k). Only dY reflects the pure
% input-to-output sensitivity.
%
% Usage
% -----
%   Params = init_params();
%   Result = cryst_open_loop_test(Params);              % defaults
%
%   % or with options:
%   opt = struct('step_scale', 1.5, 'k_step', 15, ...
%                'make_plots', true, 'verbose', true);
%   Result = cryst_open_loop_test(Params, opt);
%
% Options
% -------
%   step_scale : step size = Params.prbs_amp * step_scale  (default 1.0)
%   k_step     : sample index at which the step is applied    (default 10)
%   make_plots : true/false                                    (default true)
%   verbose    : true/false                                    (default true)
%
% Output
% ------
%   Result.t                 (Ble+1) x 1        time (min)
%   Result.Y_baseline        (Ble+1) x n_out    baseline outputs
%   Result.Y_step            (Ble+1) x n_out x n_in  per-channel outputs
%   Result.U_step            n_in x Ble x n_in  per-channel input sequences
%   Result.dY                (Ble+1) x n_out x n_in  step - baseline
%   Result.G_ss              n_in x n_out       steady-state gain (last sample)
%   Result.max_abs_dY        n_in x n_out       max |dY| over the batch
%   Result.step_amp          n_in x 1           applied step magnitude
%   Result.step_sgn          n_in x 1           step direction (+1 / -1)
%   Result.recommended_y_range  n_out x 2       heuristic [y_min, y_max]
% =========================================================================

if nargin < 2, options = struct(); end
if ~isfield(options, 'step_scale'), options.step_scale = 1.0;  end
if ~isfield(options, 'k_step'),     options.k_step     = 10;   end
if ~isfield(options, 'make_plots'), options.make_plots = true; end
if ~isfield(options, 'verbose'),    options.verbose    = true; end

Ble   = Params.Ble;
n_in  = Params.n_in;
n_out = Params.n_out;
Ts    = Params.Ts;

%% 1. Baseline run (all inputs at steady state)
U_base = repmat(Params.u_ss, 1, Ble);     % n_in x Ble
Y_base = simulate_batch_ol(Params, U_base);   % (Ble+1) x n_out

%% 2. Per-channel step runs
Result = struct();
Result.t              = (0:Ble)' * Ts;
Result.u_ss           = Params.u_ss;
Result.y_ss           = Params.y_ss;
Result.Y_baseline     = Y_base;
Result.step_amp       = zeros(n_in, 1);
Result.step_sgn       = zeros(n_in, 1);
Result.Y_step         = zeros(Ble+1, n_out, n_in);
Result.U_step         = zeros(n_in, Ble, n_in);
Result.dY             = zeros(Ble+1, n_out, n_in);
Result.G_ss           = zeros(n_in, n_out);
Result.max_abs_dY     = zeros(n_in, n_out);
Result.dY_final       = zeros(n_in, n_out);

for i = 1:n_in
    [amp, sgn] = choose_step(Params, i, options.step_scale);
    Result.step_amp(i) = amp;
    Result.step_sgn(i) = sgn;

    % Build the per-channel input trajectory
    U_step = repmat(Params.u_ss, 1, Ble);
    U_step(i, options.k_step+1:end) = Params.u_ss(i) + sgn * amp;

    % Saturation (should not trigger if choose_step is correct, but be safe)
    for k = 1:Ble
        U_step(:,k) = max(min(U_step(:,k), Params.u_phys_max), Params.u_phys_min);
    end
    Result.U_step(:,:,i) = U_step;

    % Simulate
    Y_step = simulate_batch_ol(Params, U_step);
    Result.Y_step(:,:,i) = Y_step;

    dY = Y_step - Y_base;
    Result.dY(:,:,i) = dY;
    Result.max_abs_dY(i,:) = max(abs(dY), [], 1);
    Result.dY_final(i,:)   = dY(end, :);
    Result.G_ss(i,:)       = dY(end, :) / (sgn * amp);
end

%% 3. Heuristic constraint recommendation
Result.recommended_y_range = recommend_y_range(Result, Params);

%% 4. Print summary
if options.verbose
    print_summary(Result, Params);
end

%% 5. Plots
if options.make_plots
    plot_open_loop(Result, Params);
end

end

%% =========================================================================
% Local helper: choose_step
% =========================================================================
function [amp, sgn] = choose_step(Params, i, scale)
% Pick a single-channel step magnitude that:
%   * is base_amp = Params.prbs_amp(i) * scale,
%   * respects physical bounds [u_phys_min(i), u_phys_max(i)],
%   * prefers the + direction; falls back to - if + would saturate;
%   * if neither direction fits, shrinks the amplitude to 90% of room.

    u0    = Params.u_ss(i);
    u_min = Params.u_phys_min(i);
    u_max = Params.u_phys_max(i);

    base_amp = Params.prbs_amp(i) * scale;

    pos_ok = (u0 + base_amp) <= u_max;
    neg_ok = (u0 - base_amp) >= u_min;

    if pos_ok
        sgn = +1;
    elseif neg_ok
        sgn = -1;
    else
        % Shrink to 90% of the larger room
        if (u_max - u0) >= (u0 - u_min)
            sgn = +1;
            base_amp = 0.9 * (u_max - u0);
        else
            sgn = -1;
            base_amp = 0.9 * (u0 - u_min);
        end
    end

    % Final clamp
    amp = base_amp;
    if u0 + sgn * amp > u_max
        amp = 0.95 * (u_max - u0);
    elseif u0 + sgn * amp < u_min
        amp = 0.95 * (u0 - u_min);
    end
    amp = max(amp, 0);
end

%% =========================================================================
% Local helper: simulate_batch_ol
% =========================================================================
function Y_traj = simulate_batch_ol(Params, U_seq)
    Ble   = Params.Ble;
    n_out = Params.n_out;
    Y_traj = zeros(Ble+1, n_out);
    x = Params.x0_nom;

    % k = 0 : measure + advance
    [y0, x_next] = cryst_plant(x, U_seq(:,1), Params, false);
    Y_traj(1,:) = y0';
    x = x_next;

    % k = 1 : Ble-1
    for k = 1:Ble-1
        [y, x_next] = cryst_plant(x, U_seq(:,k+1), Params, false);
        Y_traj(k+1,:) = y';
        x = x_next;
    end

    % Final measurement y(Ble); no further advance
    [yN, ~] = cryst_plant(x, U_seq(:,Ble), Params, false);
    Y_traj(Ble+1,:) = yN';
end

%% =========================================================================
% Local helper: recommend_y_range
% =========================================================================
function R = recommend_y_range(Result, Params)
    n_out = Params.n_out;
    R = zeros(n_out, 2);

    y0 = Result.Y_baseline(1, :);
    yN = Result.Y_baseline(end, :);
    drift = abs(yN - y0);
    impact = max(Result.max_abs_dY, [], 1);   % 1 x n_out

    k_drift = 1.2;
    k_step  = 3.0;

    for j = 1:n_out
        margin = k_drift*drift(j) + k_step*impact(j);
        R(j,1) = max(0, y0(j) - margin);  % 物理量通常不能为负
        R(j,2) = y0(j) + margin;
    end
end

%% =========================================================================
% Local helper: print_summary
% =========================================================================
function print_summary(Result, Params)
    n_in = Params.n_in;
    n_out = Params.n_out;

    in_names  = {'Tj_sp(K)', 'Fc(mL/s)', 'Fanti', 'Fsol', ...
                 'Nstir(rpm)', 'Fseed(g/min)'};
    out_names = {'T1(K)', 'T2(K)', 'T3(K)', 'C_ATR', 'C_NIR', ...
                 'C_rho', 'L_FBRM', 'S_turb'};

    fprintf('\n');
    fprintf('==================================================================\n');
    fprintf('  Open-loop step-response summary\n');
    fprintf('==================================================================\n\n');

    fprintf('--- Channel settings ---\n');
    for i = 1:n_in
        fprintf('  %-12s u_ss=%9.4g  step=%+9.4g  bounds=[%9.4g, %9.4g]\n', ...
            in_names{i}, Params.u_ss(i), ...
            Result.step_sgn(i)*Result.step_amp(i), ...
            Params.u_phys_min(i), Params.u_phys_max(i));
    end

    fprintf('\n--- max |dY| over the batch (delta vs baseline) ---\n');
    fprintf('%-12s', 'In \ Out');
    for j = 1:n_out, fprintf('%11s', out_names{j}); end
    fprintf('\n');
    for i = 1:n_in
        fprintf('%-12s', in_names{i});
        for j = 1:n_out, fprintf('%11.3e', Result.max_abs_dY(i,j)); end
        fprintf('\n');
    end

    fprintf('\n--- dY at the last sample ---\n');
    fprintf('%-12s', 'In \ Out');
    for j = 1:n_out, fprintf('%11s', out_names{j}); end
    fprintf('\n');
    for i = 1:n_in
        fprintf('%-12s', in_names{i});
        for j = 1:n_out, fprintf('%11.3e', Result.dY_final(i,j)); end
        fprintf('\n');
    end

    fprintf('\n--- Steady-state gain (last sample) ---\n');
    fprintf('%-12s', 'In \ Out');
    for j = 1:n_out, fprintf('%11s', out_names{j}); end
    fprintf('\n');
    for i = 1:n_in
        fprintf('%-12s', in_names{i});
        for j = 1:n_out, fprintf('%11.3e', Result.G_ss(i,j)); end
        fprintf('\n');
    end

    fprintf('\n--- Baseline natural drift (u == u_ss) ---\n');
    y0 = Result.Y_baseline(1, :);
    yN = Result.Y_baseline(end, :);
    for j = 1:n_out
        fprintf('  %-8s : y(0)=%9.4g  y(Bl)=%9.4g  drift=%+9.4g\n', ...
            out_names{j}, y0(j), yN(j), yN(j) - y0(j));
    end

    fprintf('\n--- Recommended y-constraint ranges (heuristic) ---\n');
    fprintf('  %-8s | %10s | %10s | %10s | %10s\n', ...
        'Output', 'Current min', 'Current max', 'Suggested min', 'Suggested max');
    R = Result.recommended_y_range;
    for j = 1:n_out
        fprintf('  %-8s | %10.4g | %10.4g | %10.4g | %10.4g\n', ...
            out_names{j}, Params.y_phys_min(j), Params.y_phys_max(j), ...
            R(j,1), R(j,2));
    end

    fprintf('\n=== How to use these results ===\n');
    fprintf('  1. Check the max |dY| row of the channel you intend to use as the main\n');
    fprintf('     actuator (e.g., Tj_sp for C_ATR control). If |dY| is too small, either\n');
    fprintf('     scale up Params.prbs_amp or pick a different actuator.\n');
    fprintf('  2. Review the "recommended y-constraint ranges". If the suggested range\n');
    fprintf('     is TIGHTER than the physical one, tighten Params.y_phys_min/max so MPC\n');
    fprintf('     sees the true feasible region (avoid over-aggressive moves that never\n');
    fprintf('     reach the reference).\n');
    fprintf('  3. Use the drift row to verify that at u_ss the tracked output does not\n');
    fprintf('     escape the physical bounds on its own. If it does, move Ref_points.\n');
    fprintf('  4. Adjust Params.Kp / Params.y_ref_phys so that the trajectory is reachable\n');
    fprintf('     with the actuators given the gains above.\n');
    fprintf('==================================================================\n');
end

%% =========================================================================
% Local helper: plot_open_loop
% =========================================================================
function plot_open_loop(Result, Params)
    n_in = Params.n_in;
    n_out = Params.n_out;
    Ble = Params.Ble;
    t = Result.t;

    in_names  = {'T_{j,sp}(K)', 'F_c(mL/s)', 'F_{anti}(mL/s)', ...
                 'F_{sol}(mL/s)', 'N_{stir}(rpm)', 'F_{seed}(g/min)'};
    out_names = {'T_1(K)', 'T_2(K)', 'T_3(K)', 'C_{ATR}', 'C_{NIR}', ...
                 'C_{\rho}', 'L_{FBRM}(\mum)', 'S_{turb}(NTU)'};

    %% Figure 1: input excitations
    figure('Name', 'Open-loop inputs', 'Color', 'w');
    for i = 1:n_in
        subplot(2, 3, i); hold on; grid on;
        plot(t(1:Ble), Result.U_step(i, :, i), 'b-', 'LineWidth', 1.8);
        plot([t(1), t(Ble)], [Params.u_phys_max(i), Params.u_phys_max(i)], 'k--', 'LineWidth', 0.8);
        plot([t(1), t(Ble)], [Params.u_phys_min(i), Params.u_phys_min(i)], 'k--', 'LineWidth', 0.8, 'HandleVisibility', 'off');
        xlabel('Time (min)'); ylabel(in_names{i});
        title(sprintf('Channel %d', i));
        xlim([t(1), t(Ble)]);
    end

    %% Figures 2..n_in+1: one figure per input, all outputs
    for i = 1:n_in
        figure('Name', sprintf('Open-loop response: channel %d', i), 'Color', 'w');
        for j = 1:n_out
            subplot(4, 2, j); hold on; grid on;
            plot(t, Result.Y_baseline(:,j), 'k--', 'LineWidth', 1.2);
            plot(t, Result.Y_step(:,j,i),   'b-',  'LineWidth', 1.6);
            title(sprintf('%s | step %+g on ch.%d', out_names{j}, ...
                Result.step_sgn(i)*Result.step_amp(i), i), 'FontSize', 9);
            xlabel('Time (min)'); xlim([t(1), t(end)]);
            if j == 1
                legend('Baseline', 'Step', 'Location', 'best', 'FontSize', 7);
            end
        end
    end

    %% Figure n_in+2: dY envelope (each output one subplot, all channels)
    figure('Name', 'Open-loop dY envelope', 'Color', 'w');
    colors = lines(n_in);
    for j = 1:n_out
        subplot(4, 2, j); hold on; grid on;
        for i = 1:n_in
            plot(t, Result.dY(:,j,i), '-', 'Color', colors(i,:), 'LineWidth', 1.4);
        end
        title(out_names{j}, 'FontSize', 9);
        xlabel('Time (min)');
        if j == 1
            legend(in_names, 'Location', 'best', 'FontSize', 6);
        end
        xlim([t(1), t(end)]);
    end
end
