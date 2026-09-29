function plot_results(Data, Params, varargin)
% =========================================================================
% plot_results -- Comprehensive visualization for DyPLS-IL-MPC simulations.
%
% -------------------------------------------------------------------------
% Figures produced (in order)
% -------------------------------------------------------------------------
%   Fig 1 : Output tracking      -- y_1 only (for clarity), vs bounds/reference
%   Fig 2 : Control inputs       -- v_1 and v_2 vs. bounds
%   Fig 3 : ILC convergence      -- batch-to-batch error norm
%   Fig 4 : DyPLS-ARX fidelity   -- standardized output reconstruction
%   Fig 5 : Final batch          -- actual vs. nominal vs. reference
%   Fig 6 : Tube diagnostics     -- printed text only
%
% -------------------------------------------------------------------------
% Axis convention (unified to sample index k, NOT time in minutes)
% -------------------------------------------------------------------------
%   Fig 1  : x = 1 : Params.Ble
%   Fig 2  : x = 1 : Params.Ble-1
%   Fig 3  : x = 1 : Params.N_batches
%   Fig 4  : unchanged
%   Fig 5  : subplot 1,2 : x = 1 : Params.Ble
%            subplot 3   : x = 1 : Params.Ble-1
% =========================================================================

    %% ==================== 0. Plot switch ====================
    if isfield(Params, 'plot_results') && ~isempty(Params.plot_results) ...
            && ~Params.plot_results
        fprintf('plot_results: Params.plot_results = false; skipping.\n');
        return;
    end

    %% ==================== 1. Optional arguments ====================
    Model = [];
    Tube  = [];

    if numel(varargin) >= 1 && ~isempty(varargin{1})
        Model = varargin{1};
    end
    if numel(varargin) >= 2 && ~isempty(varargin{2})
        Tube = varargin{2};
    end

    %% ==================== 2. Unpack data ====================
    N         = Params.Ble;
    N_batches = Params.N_batches;
    n_out     = Params.n_out;
    n_in      = Params.n_in;

    Y_hist         = Data.Y_hist;
    V_hist         = Data.V_hist;
    Y_nominal_hist = Data.Y_nominal_hist;
    V_nominal_hist = Data.V_nominal_hist;
    Err_actual     = Data.Error_norm_actual(:);
    Err_nominal    = Data.Error_norm_nominal(:);

    ub_y = Params.ub_y(:);   lb_y = Params.lb_y(:);
    ub_v = Params.ub_v(:);   lb_v = Params.lb_v(:);

    Kp         = Params.Kp(:);
    Ref_points = Params.Ref_points(:);

    %% ==================== 3. Dimension sanity checks ====================
    assert(size(Y_hist, 1) == n_out, ...
        'plot_results: Y_hist rows (%d) != n_out (%d).', ...
        size(Y_hist, 1), n_out);
    assert(size(Y_hist, 2) == N, ...
        'plot_results: Y_hist columns (%d) != Ble (%d).', ...
        size(Y_hist, 2), N);
    assert(size(Y_hist, 3) == N_batches, ...
        'plot_results: Y_hist pages (%d) != N_batches (%d).', ...
        size(Y_hist, 3), N_batches);
    assert(size(V_hist, 2) == N - 1, ...
        'plot_results: V_hist columns (%d) != Ble-1 (%d).', ...
        size(V_hist, 2), N - 1);
    assert(size(Y_nominal_hist, 3) == N_batches + 1, ...
        'plot_results: Y_nominal_hist pages (%d) != N_batches+1 (%d).', ...
        size(Y_nominal_hist, 3), N_batches + 1);
    assert(numel(Kp) == numel(Ref_points), ...
        'plot_results: Kp and Ref_points must have equal length.');

    %% ==================== 4. Sample-index vectors ====================
    t_y = (1:N)';        % for outputs:    k = 1..Ble
    t_v = (1:N - 1)';    % for inputs :    k = 1..Ble-1

    %% ==================== 5. Batch selection ====================
    idx_y = [1, 3, 5, 7, 10, 15];
    idx_y = idx_y(idx_y >= 1 & idx_y <= N_batches);
    if isempty(idx_y), idx_y = 1; end

    idx_v = [1, 3, 7, 15];
    idx_v = idx_v(idx_v >= 1 & idx_v <= N_batches);
    if isempty(idx_v), idx_v = 1; end

    %% ==================== 6. Plot styling ====================
    colors = [ 0.85 0.10 0.10;    % red
               0.00 0.20 0.80;    % blue
               0.00 0.60 0.00;    % green
               0.80 0.00 0.80;    % magenta
               0.95 0.55 0.00;    % orange
               0.00 0.60 0.60 ];  % teal

    lw_nom = 1.0;
    lw_act = 1.5;
    lw_bnd = 0.8;
    ms_ref = 4;
    lw_ref = 1.0;
    fs_leg = 8;

%% =====================================================================
% Figure 1 -- Output tracking (y_1), x = 1 : Ble
% =====================================================================
figure('Name', 'Output Tracking', 'Color', 'w');

n_out_plot = 1;
n_show_y   = numel(idx_y);

for i = 1:n_out_plot
    subplot(n_out_plot, 1, i);
    hold on; grid on;

    h_bnd_hi = plot(t_y, ub_y(i) * ones(N, 1), 'k--', 'LineWidth', lw_bnd);
    h_bnd_lo = plot(t_y, lb_y(i) * ones(N, 1), 'k--', 'LineWidth', lw_bnd);
    set(h_bnd_lo, 'HandleVisibility', 'off');

    h_ref = [];
    if ~isempty(Kp)
        for j = 1:numel(Kp)
            plot([Kp(j) Kp(j)], [lb_y(i) ub_y(i)], ':', ...
                'Color', [0.80 0.00 0.00], 'LineWidth', 0.5, ...
                'HandleVisibility', 'off');
        end
        h_ref = plot(Kp, Ref_points, 'o', ...
            'MarkerSize',        ms_ref, ...
            'MarkerFaceColor',   'none', ...
            'MarkerEdgeColor',   'r', ...
            'LineWidth',         lw_ref);
    end

    h_batch = zeros(1, n_show_y);
    for j = 1:n_show_y
        b   = idx_y(j);
        clr = colors(mod(j - 1, size(colors, 1)) + 1, :);

        plot(t_y, Y_nominal_hist(i, :, b + 1), '--', ...
            'Color', clr, 'LineWidth', lw_nom, ...
            'HandleVisibility', 'off');

        h_batch(j) = plot(t_y, Y_hist(i, :, b), '-', ...
            'Color', clr, 'LineWidth', lw_act);
    end

    xlabel('sample k');
    ylabel(sprintf('y_%d', i));
    xlim([1, N]);                       % <-- enforce k = 1..Ble

    leg_handles = h_batch;
    leg_labels  = cell(1, n_show_y);
    for j = 1:n_show_y
        leg_labels{j} = sprintf('Batch %d', idx_y(j));
    end

    if ~isempty(h_ref)
        leg_handles = [leg_handles, h_ref];
        leg_labels  = [leg_labels,  'Reference'];
    else
        leg_handles = [leg_handles, h_bnd_hi];
        leg_labels  = [leg_labels,  'Constraint'];
    end

    h_leg = legend(leg_handles, leg_labels, ...
        'Location',    'southeast', ...
        'Orientation', 'vertical');
    set(h_leg, 'Box', 'on', 'FontSize', fs_leg);
end

%% =====================================================================
% Figure 2 -- Control inputs, x = 1 : Ble-1
% =====================================================================
figure('Name', 'Control Inputs', 'Color', 'w');

n_show_v = numel(idx_v);

for i = 1:n_in
    subplot(n_in, 1, i);
    hold on; grid on;

    h_bnd_hi = plot(t_v, ub_v(i) * ones(N - 1, 1), 'k--', ...
        'LineWidth', lw_bnd);
    h_bnd_lo = plot(t_v, lb_v(i) * ones(N - 1, 1), 'k--', ...
        'LineWidth', lw_bnd);
    set(h_bnd_lo, 'HandleVisibility', 'off');

    h_batch = zeros(1, n_show_v);
    for j = 1:n_show_v
        b   = idx_v(j);
        clr = colors(mod(j - 1, size(colors, 1)) + 1, :);

        plot(t_v, V_nominal_hist(i, :, b + 1), '--', ...
            'Color', clr, 'LineWidth', lw_nom, ...
            'HandleVisibility', 'off');

        h_batch(j) = plot(t_v, V_hist(i, :, b), '-', ...
            'Color', clr, 'LineWidth', lw_act);
    end

    xlabel('sample k');
    ylabel(sprintf('v_%d', i));
    xlim([1, N - 1]);                   % <-- enforce k = 1..Ble-1

    leg_handles = h_batch;
    leg_labels  = cell(1, n_show_v);
    for j = 1:n_show_v
        leg_labels{j} = sprintf('Batch %d', idx_v(j));
    end

    leg_handles = [leg_handles, h_bnd_hi];
    leg_labels  = [leg_labels,  'Constraint'];

    h_leg = legend(leg_handles, leg_labels, ...
        'Location',    'south', ...
        'Orientation', 'horizontal');
    set(h_leg, 'Box', 'on', 'FontSize', fs_leg);

    yl = ylim();
    dy = 0.25 * (yl(2) - yl(1));
    ylim([yl(1) - dy, yl(2)]);
end

%% =====================================================================
% Figure 3 -- ILC convergence, x = 1 : N_batches
% =====================================================================
figure('Name', 'ILC Convergence', 'Color', 'w');
hold on;
grid on;

batch_id = (1:N_batches)';

h1 = plot(batch_id, Err_actual,  'b-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7, ...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
h2 = plot(batch_id, Err_nominal, 'r-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');

set(gca, 'YScale', 'linear');

% Exact axis range on integer batch indices
xlim([1, N_batches]);
set(gca, 'XTick', 1:N_batches);         % <-- integer ticks only

xlabel('batch number');
ylabel('Error norm at reference points');
legend([h1, h2], {'Actual error', 'Nominal error'}, ...
    'Location', 'best');

fprintf('plot_results: ILC error norm summary\n');
fprintf('  batch 1          : nominal = %.4e, actual = %.4e\n', ...
    Err_nominal(1), Err_actual(1));
fprintf('  batch %-3d       : nominal = %.4e, actual = %.4e\n', ...
    N_batches, Err_nominal(end), Err_actual(end));

%% =====================================================================
% Figure 4 -- DyPLS-ARX model fidelity (unchanged)
% =====================================================================
if ~isempty(Model) && isfield(Model, 'U') && ...
        isfield(Model, 'Q') && isfield(Model, 'R_y')

    Y0_hat = Model.U * Model.Q';
    Y0     = Y0_hat + Model.R_y;

    n_ch = size(Y0, 2);
    t_m  = (1:size(Y0, 1))';

    figure('Name', 'DyPLS-ARX Fidelity', 'Color', 'w');

    for i = 1:n_ch
        subplot(n_ch, 1, i);
        hold on;
        grid on;

        plot(t_m, Y0(:, i),     'b-',  'LineWidth', 1.0);
        plot(t_m, Y0_hat(:, i), 'r--', 'LineWidth', 1.0);

        xlabel('time (min)');
        ylabel(sprintf('Standardized y_%d', i));
        legend('True standardized', 'Model reconstruction', ...
            'Location', 'best');

        rel_err = norm(Y0(:, i) - Y0_hat(:, i)) ...
                / max(norm(Y0(:, i)), eps);
        fprintf(['plot_results: DyPLS-ARX relative error ' ...
                 '(channel %d) = %.4e\n'], i, rel_err);
    end
end

%% =====================================================================
% Figure 5 -- Final batch
%   subplot 1,2 : x = 1 : Ble
%   subplot 3   : x = 1 : Ble-1
% =====================================================================
Y_batch_last         = Y_hist(:, :, N_batches);
Y_nominal_batch_last = Y_nominal_hist(:, :, N_batches + 1);
V_batch_last         = V_hist(:, :, N_batches);
V_nominal_batch_last = V_nominal_hist(:, :, N_batches + 1);

Kp_vec  = Params.Kp(:);
Ref_vec = Params.Ref_points(:);

% Sample-index axes
t_y2 = (1:N)';
t_v2 = (1:N - 1)';

lw_traj = 1.2;
ms_ref  = 4.5;
lw_ref  = 1.0;
fs_leg  = 8;

figure('Name', sprintf('Final Batch %d', N_batches), 'Color', 'w');

% ---------------------------------------------------------------------
% Subplot 1 -- y_1 with references, x = 1 : Ble
% ---------------------------------------------------------------------
subplot(3, 1, 1);
hold on; grid on;

h_y1_act = plot(t_y2, Y_batch_last(1, :),         'b-',  ...
    'LineWidth', lw_traj);
h_y1_nom = plot(t_y2, Y_nominal_batch_last(1, :), 'r--', ...
    'LineWidth', lw_traj);
h_ref    = plot(Kp_vec, Ref_vec, 'o', ...
    'MarkerSize',       ms_ref, ...
    'LineWidth',        lw_ref, ...
    'MarkerEdgeColor', 'r', ...
    'MarkerFaceColor', 'none');

xlabel('sample k');
ylabel('y_1');
xlim([1, N]);                       % <-- enforce k = 1..Ble

h_leg1 = legend([h_y1_act, h_y1_nom], ...
    {'actual y_1', 'nominal y_1'}, ...
    'Location',    'northeast', ...
    'Orientation', 'vertical');
set(h_leg1, 'Box', 'on', 'FontSize', fs_leg);

% ---------------------------------------------------------------------
% Subplot 2 -- y_2, x = 1 : Ble
% ---------------------------------------------------------------------
subplot(3, 1, 2);
hold on; grid on;

h_y2_act = plot(t_y2, Y_batch_last(2, :),         'b-',  ...
    'LineWidth', lw_traj);
h_y2_nom = plot(t_y2, Y_nominal_batch_last(2, :), 'r--', ...
    'LineWidth', lw_traj);

xlabel('sample k');
ylabel('y_2');
xlim([1, N]);                       % <-- enforce k = 1..Ble

h_leg2 = legend([h_y2_act, h_y2_nom], ...
    {'actual y_2', 'nominal y_2'}, ...
    'Location',    'northeast', ...
    'Orientation', 'vertical');
set(h_leg2, 'Box', 'on', 'FontSize', fs_leg);

% ---------------------------------------------------------------------
% Subplot 3 -- control inputs, x = 1 : Ble-1
% ---------------------------------------------------------------------
subplot(3, 1, 3);
hold on; grid on;

h_v1_act = plot(t_v2, V_batch_last(1, :),         'b-',  ...
    'LineWidth', lw_traj);
h_v1_nom = plot(t_v2, V_nominal_batch_last(1, :), 'r--', ...
    'LineWidth', lw_traj);
h_v2_act = plot(t_v2, V_batch_last(2, :),         'g-',  ...
    'LineWidth', lw_traj);
h_v2_nom = plot(t_v2, V_nominal_batch_last(2, :), 'm--', ...
    'LineWidth', lw_traj);

xlabel('sample k');
ylabel('v');
xlim([1, N - 1]);                   % <-- enforce k = 1..Ble-1

h_leg3 = legend([h_v1_act, h_v2_act], ...
    {'v_1 actual', 'v_2 actual'}, ...
    'Location',    'northeast', ...
    'Orientation', 'vertical');
set(h_leg3, 'Box', 'on', 'FontSize', fs_leg);

%% =====================================================================
% Figure 6 -- Tube diagnostics (text only)
% =====================================================================
if ~isempty(Tube)
    fprintf('\nplot_results: Tube diagnostics\n');

    if isfield(Tube, 'spectral_radius')
        fprintf('  Closed-loop spectral radius : %.6f\n', ...
            Tube.spectral_radius);
    end
    if isfield(Tube, 'P_horizon')
        fprintf('  Prediction horizon          : %d\n', ...
            Tube.P_horizon);
    end
    if isfield(Tube, 'delta_v')
        fprintf('  Input  delta_v              : [%s]\n', ...
            num2str(Tube.delta_v(:)', '%.3e '));
    end
    if isfield(Tube, 'delta_y')
        fprintf('  Output delta_y              : [%s]\n', ...
            num2str(Tube.delta_y(:)', '%.3e '));
    end
    if isfield(Tube, 'h_v_tight') && any(Tube.h_v_tight < 0)
        warning('plot_results: h_v_tight has negative entries.');
    end
    if isfield(Tube, 'h_y_tight') && any(Tube.h_y_tight < 0)
        warning('plot_results: h_y_tight has negative entries.');
    end
end

fprintf('plot_results: plotting completed.\n');
end
