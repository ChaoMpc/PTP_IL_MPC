function plot_results(Data, Params, varargin)
% =========================================================================
% plot_results -- Visualization for DyPLS-IL-MPC simulations.
%
% Deviation-to-physical conversion
% --------------------------------
%   All quantities stored in Data (Y_hist, V_hist, Y_nominal_hist,
%   V_nominal_hist) are PHYSICAL DEVIATIONS from the steady state:
%
%       Y_hist(i,k,b)  = Delta y_i(k) at batch b
%       V_hist(i,k,b)  = Delta v_i(k) at batch b
%
%   To display physically meaningful values, the steady state is added:
%
%       y_i(k,b) = Params.y_ss(i) + Y_hist(i,k,b)
%       v_i(k,b) = Params.u_ss(i) + V_hist(i,k,b)
%
%   The reference points Params.y_ref_phys are already absolute physical
%   values, and the constraint bounds Params.y_phys_min/max,
%   Params.u_phys_min/max are also absolute physical. No further
%   conversion is needed for them.
%
% Figures produced
% ----------------
%   Fig 1 : Output tracking      -- y_1 and y_2 (physical), vs. bounds
%   Fig 2 : Control inputs       -- v_1 and v_2 (physical), vs. bounds
%   Fig 3 : ILC convergence      -- batch-to-batch error norm
%   Fig 4 : DyPLS-ARX fidelity   -- standardized deviation reconstruction
%   Fig 5 : Final batch          -- actual vs. nominal vs. reference
%
% Calling convention
% ------------------
%   plot_results(Data, Params)
%   plot_results(Data, Params, Model)
%   plot_results(Data, Params, Model, Tube)
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
    Ts        = Params.Ts;

    if isfield(Data, 'y_ss') && ~isempty(Data.y_ss)
        y_ss = Data.y_ss(:);
    else
        y_ss = Params.y_ss(:);
    end
    if isfield(Data, 'u_ss') && ~isempty(Data.u_ss)
        u_ss = Data.u_ss(:);
    else
        u_ss = Params.u_ss(:);
    end

    Y_dev_hist         = Data.Y_hist;
    V_dev_hist         = Data.V_hist;
    Y_dev_nominal_hist = Data.Y_nominal_hist;
    V_dev_nominal_hist = Data.V_nominal_hist;

    Err_actual  = Data.Error_norm_actual(:);
    Err_nominal = Data.Error_norm_nominal(:);

    y_phys_max = Params.y_phys_max(:);
    y_phys_min = Params.y_phys_min(:);
    u_phys_max = Params.u_phys_max(:);
    u_phys_min = Params.u_phys_min(:);

    Kp             = Params.Kp(:);
    y_ref_phys     = Params.y_ref_phys(:);

    % ---- Convert deviation data to physical absolute values ---------
    Y_hist_phys         = bsxfun(@plus, Y_dev_hist,         reshape(y_ss, [n_out, 1, 1]));
    Y_nominal_hist_phys = bsxfun(@plus, Y_dev_nominal_hist, reshape(y_ss, [n_out, 1, 1]));
    V_hist_phys         = bsxfun(@plus, V_dev_hist,         reshape(u_ss, [n_in,  1, 1]));
    V_nominal_hist_phys = bsxfun(@plus, V_dev_nominal_hist, reshape(u_ss, [n_in,  1, 1]));

    %% ==================== 3. Dimension sanity checks ====================
    assert(size(Y_hist_phys, 1) == n_out, ...
        'plot_results: Y_hist rows (%d) != n_out (%d).', ...
        size(Y_hist_phys, 1), n_out);
    assert(size(Y_hist_phys, 2) == N, ...
        'plot_results: Y_hist columns (%d) != Ble (%d).', ...
        size(Y_hist_phys, 2), N);
    assert(size(Y_hist_phys, 3) == N_batches, ...
        'plot_results: Y_hist pages (%d) != N_batches (%d).', ...
        size(Y_hist_phys, 3), N_batches);
    assert(size(V_hist_phys, 2) == N - 1, ...
        'plot_results: V_hist columns (%d) != Ble-1 (%d).', ...
        size(V_hist_phys, 2), N - 1);
    assert(numel(Kp) == numel(y_ref_phys), ...
        'plot_results: Kp and y_ref_phys must have equal length.');

    %% ==================== 4. Time vectors ====================
    % X-axis uses the 1-based sample index k = 1 .. N so that the
    % reference markers (Kp = [20, 40, 60]) line up exactly with the
    % plotted sample positions. This applies to Figure 1, Figure 2 and
    % Figure 5, all of which reuse t_y and t_v.
    t_y = (1:N)'     * Ts;      % 1 .. N
    t_v = (1:N - 1)' * Ts;      % 1 .. N-1

    %% ==================== 5. Plot styling ====================
    colors = [ 0.000 0.447 0.741;
               0.850 0.325 0.098;
               0.929 0.694 0.125;
               0.494 0.184 0.556;
               0.466 0.674 0.188;
               0.301 0.745 0.933 ];

    lw_nom = 1.0;
    lw_act = 1.6;
    lw_bnd = 0.8;
    ms_ref = 5;
    fs_leg = 8;

%% =====================================================================
% Figure 1 -- Output tracking (physical y_1, y_2)
% =====================================================================
figure('Name', 'Output Tracking', 'Color', 'w');

desired_batches = [1, 2, 3, 10];
batch_list      = desired_batches(desired_batches <= N_batches);
if isempty(batch_list)
    batch_list = 1;
end
n_show = numel(batch_list);

for i = 1:n_out
    subplot(n_out, 1, i);
    hold on; grid on;

    h_bnd_hi = plot(t_y, y_phys_max(i) * ones(N, 1), 'k--', 'LineWidth', lw_bnd);
    h_bnd_lo = plot(t_y, y_phys_min(i) * ones(N, 1), 'k--', 'LineWidth', lw_bnd);
    set(h_bnd_lo, 'HandleVisibility', 'off');

    h_ref = [];
    if i == 1 && ~isempty(Kp)
        for j = 1:numel(Kp)
            plot([t_y(Kp(j)) t_y(Kp(j))], [y_phys_min(i) y_phys_max(i)], ':', ...
                'Color', [0.80 0.00 0.00], 'LineWidth', 0.5, ...
                'HandleVisibility', 'off');
        end
        h_ref = plot(t_y(Kp), y_ref_phys, 'o', ...
            'MarkerSize',        ms_ref, ...
            'MarkerFaceColor',   'none', ...
            'MarkerEdgeColor',   'r', ...
            'LineWidth',         1.0);
    end

    h_batch = zeros(1, n_show);
    for j = 1:n_show
        b   = batch_list(j);
        clr = colors(mod(j - 1, size(colors, 1)) + 1, :);

        plot(t_y, Y_nominal_hist_phys(i, :, b + 1), '--', ...
            'Color', clr, 'LineWidth', lw_nom, ...
            'HandleVisibility', 'off');

        h_batch(j) = plot(t_y, Y_hist_phys(i, :, b), '-', ...
            'Color', clr, 'LineWidth', lw_act);
    end

    xlabel('Sample k');
    ylabel(sprintf('y_%d (wt%%)', i));

    leg_handles = h_batch;
    leg_labels  = cell(1, n_show);
    for j = 1:n_show
        leg_labels{j} = sprintf('Batch %d', batch_list(j));
    end

    if i == 1 && ~isempty(h_ref)
        leg_handles = [leg_handles, h_ref];
        leg_labels  = [leg_labels,  'Reference'];
    else
        leg_handles = [leg_handles, h_bnd_hi];
        leg_labels  = [leg_labels,  'Constraint'];
    end

    h_leg = legend(leg_handles, leg_labels, ...
        'Location',    'south', ...
        'Orientation', 'horizontal');
    set(h_leg, 'Box', 'on', 'FontSize', fs_leg);

    yl = ylim();
    dy = 0.25 * (yl(2) - yl(1));
    ylim([yl(1) - dy, yl(2)]);

    xlim([1, Params.Ble]);
end

%% =====================================================================
% Figure 2 -- Control inputs (physical v_1, v_2)
% =====================================================================
figure('Name', 'Control Inputs', 'Color', 'w');

for i = 1:n_in
    subplot(n_in, 1, i);
    hold on; grid on;

    h_bnd_hi = plot(t_v, u_phys_max(i) * ones(N - 1, 1), 'k--', ...
        'LineWidth', lw_bnd);
    h_bnd_lo = plot(t_v, u_phys_min(i) * ones(N - 1, 1), 'k--', ...
        'LineWidth', lw_bnd);
    set(h_bnd_lo, 'HandleVisibility', 'off');

    h_batch = zeros(1, n_show);
    for j = 1:n_show
        b   = batch_list(j);
        clr = colors(mod(j - 1, size(colors, 1)) + 1, :);

        plot(t_v, V_nominal_hist_phys(i, :, b + 1), '--', ...
            'Color', clr, 'LineWidth', lw_nom, ...
            'HandleVisibility', 'off');

        h_batch(j) = plot(t_v, V_hist_phys(i, :, b), '-', ...
            'Color', clr, 'LineWidth', lw_act);
    end

    xlabel('Sample k');
    if i == 1
        ylabel('v_1 = R (lb/min)');
    else
        ylabel('v_2 = S (lb/min)');
    end

    leg_handles = h_batch;
    leg_labels  = cell(1, n_show);
    for j = 1:n_show
        leg_labels{j} = sprintf('Batch %d', batch_list(j));
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

    xlim([1, Params.Ble - 1]);
end

%% =====================================================================
% Figure 3 -- ILC batch-to-batch convergence
% =====================================================================
figure('Name', 'ILC Convergence', 'Color', 'w');
hold on; grid on;

batch_id = (1:N_batches)';

h1 = plot(batch_id, Err_actual,  'b-o', ...
    'LineWidth', 1.5, 'MarkerSize', 7, ...
    'MarkerFaceColor', 'b', 'MarkerEdgeColor', 'k');
h2 = plot(batch_id, Err_nominal, 'r-s', ...
    'LineWidth', 1.5, 'MarkerSize', 7, ...
    'MarkerFaceColor', 'r', 'MarkerEdgeColor', 'k');

set(gca, 'YScale', 'linear');

% ---- Fixed x-axis: batch index from 1 to N_batches ----
xlim([1, N_batches]);
xticks(1:N_batches);
xticklabels(arrayfun(@(b) sprintf('%d', b), 1:N_batches, ...
    'UniformOutput', false));

xlabel('Batch number');
ylabel('Error norm at reference points');
legend([h1, h2], {'Actual error', 'Nominal error'}, 'Location', 'best');

fprintf('plot_results: ILC error norm summary\n');
fprintf('  batch 1          : nominal = %.4e, actual = %.4e\n', ...
    Err_nominal(1), Err_actual(1));
fprintf('  batch %-3d       : nominal = %.4e, actual = %.4e\n', ...
    N_batches, Err_nominal(end), Err_actual(end));

%% =====================================================================
% Figure 4 -- DyPLS-ARX fidelity (standardized deviation space)
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
        hold on; grid on;

        plot(t_m, Y0(:, i),     'b-',  'LineWidth', 1.0);
        plot(t_m, Y0_hat(:, i), 'r--', 'LineWidth', 1.0);

        xlabel('Sample k');
        ylabel(sprintf('Standardized y_%d', i));
        legend('True standardized', 'Model reconstruction', 'Location', 'best');

        rel_err = norm(Y0(:, i) - Y0_hat(:, i)) ...
                / max(norm(Y0(:, i)), eps);
        fprintf(['plot_results: DyPLS-ARX relative error ' ...
                 '(channel %d) = %.4e\n'], i, rel_err);
    end
end

%% =====================================================================
% Figure 5 -- Final batch (physical)
% =====================================================================
Y_batch_last         = Y_hist_phys(:, :, N_batches);
Y_nominal_batch_last = Y_nominal_hist_phys(:, :, N_batches + 1);
V_batch_last         = V_hist_phys(:, :, N_batches);
V_nominal_batch_last = V_nominal_hist_phys(:, :, N_batches + 1);

lw_traj = 1.2;
ms_ref  = 4.5;
dy_frac = 0.30;

figure('Name', sprintf('Final Batch %d', N_batches), 'Color', 'w');

% Subplot 1 -- physical y1 with references
subplot(3, 1, 1);
hold on; grid on;

h_y1_act = plot(t_y, Y_batch_last(1, :),         'b-',  'LineWidth', lw_traj);
h_y1_nom = plot(t_y, Y_nominal_batch_last(1, :), 'r--', 'LineWidth', lw_traj);
h_ref    = plot(t_y(Kp), y_ref_phys, 'ko', ...
    'MarkerSize',       ms_ref, ...
    'LineWidth',        1.0, ...
    'MarkerFaceColor', 'none');

xlabel('Sample k');
ylabel('y_1 (wt%)');

h_leg1 = legend([h_y1_act, h_y1_nom, h_ref], ...
    {'actual y_1', 'nominal y_1', 'reference'}, ...
    'Location', 'southeast', 'Orientation', 'horizontal');
set(h_leg1, 'Box', 'on', 'FontSize', fs_leg);

yl = ylim();
dy = dy_frac * (yl(2) - yl(1));
ylim([yl(1) - dy, yl(2)]);
xlim([1, Params.Ble]);

% Subplot 2 -- physical y2 (no reference imposed)
subplot(3, 1, 2);
hold on; grid on;

h_y2_act = plot(t_y, Y_batch_last(2, :),         'b-',  'LineWidth', lw_traj);
h_y2_nom = plot(t_y, Y_nominal_batch_last(2, :), 'r--', 'LineWidth', lw_traj);

xlabel('Sample k');
ylabel('y_2 (wt%)');

h_leg2 = legend([h_y2_act, h_y2_nom], ...
    {'actual y_2', 'nominal y_2'}, ...
    'Location', 'southeast', 'Orientation', 'horizontal');
set(h_leg2, 'Box', 'on', 'FontSize', fs_leg);

yl = ylim();
dy = dy_frac * (yl(2) - yl(1));
ylim([yl(1) - dy, yl(2)]);
xlim([1, Params.Ble]);

% Subplot 3 -- physical control inputs
subplot(3, 1, 3);
hold on; grid on;

h_v1_act = plot(t_v, V_batch_last(1, :),         'b-',  'LineWidth', lw_traj);
h_v1_nom = plot(t_v, V_nominal_batch_last(1, :), 'r--', 'LineWidth', lw_traj);
h_v2_act = plot(t_v, V_batch_last(2, :),         'g-',  'LineWidth', lw_traj);
h_v2_nom = plot(t_v, V_nominal_batch_last(2, :), 'm--', 'LineWidth', lw_traj);

xlabel('Sample k');
ylabel('v (lb/min)');

h_leg3 = legend([h_v1_act, h_v1_nom, h_v2_act, h_v2_nom], ...
    {'v_1 actual', 'v_1 nominal', 'v_2 actual', 'v_2 nominal'}, ...
    'Location', 'east', 'Orientation', 'horizontal');
set(h_leg3, 'Box', 'on', 'FontSize', fs_leg);

yl = ylim();
dy = dy_frac * (yl(2) - yl(1));
ylim([yl(1) - dy, yl(2)]);

xlim([1, Params.Ble - 1]);

%% =====================================================================
% Figure 6 -- Tube diagnostics (text only)
% =====================================================================
if ~isempty(Tube)
    fprintf('\nplot_results: Tube diagnostics\n');

    if isfield(Tube, 'spectral_radius')
        fprintf('  Closed-loop spectral radius : %.6f\n', Tube.spectral_radius);
    end
    if isfield(Tube, 'P_horizon')
        fprintf('  Prediction horizon          : %d\n', Tube.P_horizon);
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

%% =====================================================================
% Tracking error table -- per-batch errors at the reference points
% ---------------------------------------------------------------------
% Rows    : batch index b = 1..N_batches
% Columns :
%   column 1     : batch index
%   columns 2-4  : actual  y1 tracking error at Kp(1), Kp(2), Kp(3)
%   columns 5-7  : nominal y1 tracking error at Kp(1), Kp(2), Kp(3)
%   column  8    : total actual  tracking error (2-norm over Kp)
%   column  9    : total nominal tracking error (2-norm over Kp)
%
% All errors are in PHYSICAL DEVIATION units [wt%].
% =====================================================================
n_ref = numel(Params.Kp);

E_act_pt = zeros(N_batches, n_ref);
E_nom_pt = zeros(N_batches, n_ref);

for b = 1:N_batches
    for j = 1:n_ref
        k_j = Params.Kp(j);

        % Data.Y_hist             : n_out x N x N_batches
        % Data.Y_nominal_hist     : n_out x N x (N_batches + 1)
        %   -> page b+1 holds the nominal trajectory AFTER batch b
        E_act_pt(b, j) = Y_dev_hist(1, k_j, b) ...
                         - Params.Ref_points(j);
        E_nom_pt(b, j) = Y_dev_nominal_hist(1, k_j, b + 1) ...
                         - Params.Ref_points(j);
    end
end

E_act_total = sqrt(sum(E_act_pt.^2, 2));
E_nom_total = sqrt(sum(E_nom_pt.^2, 2));

% ---- Package into a structure ---------------------------------------
TrackingErrorTable = struct();
TrackingErrorTable.batch       = (1:N_batches)';
TrackingErrorTable.Kp          = Params.Kp(:);
TrackingErrorTable.Ref_points  = Params.Ref_points(:);
TrackingErrorTable.E_act_pt    = E_act_pt;
TrackingErrorTable.E_nom_pt    = E_nom_pt;
TrackingErrorTable.E_act_total = E_act_total;
TrackingErrorTable.E_nom_total = E_nom_total;
TrackingErrorTable.units       = 'wt% (physical deviation)';

% ---- Numeric matrix (batch, [e_act_Kp*], [e_nom_Kp*], totals) -------
TrackingErrorMatrix = [ (1:N_batches)', E_act_pt, E_nom_pt, ...
                        E_act_total, E_nom_total ];

col_names = cell(1, size(TrackingErrorMatrix, 2));
col_names{1} = 'batch';
for j = 1:n_ref
    col_names{1 + j}         = sprintf('e_act_Kp%d', j);
    col_names{1 + n_ref + j} = sprintf('e_nom_Kp%d', j);
end
col_names{end - 1} = 'E_act_total';
col_names{end}     = 'E_nom_total';

% ---- Print a readable table in the console --------------------------
fprintf('\nplot_results: tracking error table (physical deviation, wt%%)\n');
fprintf('%6s', 'batch');
for j = 1:n_ref
    fprintf('%14s', sprintf('act Kp%d', j));
end
for j = 1:n_ref
    fprintf('%14s', sprintf('nom Kp%d', j));
end
fprintf('%16s%16s\n', 'act total', 'nom total');

for b = 1:N_batches
    fprintf('%6d', b);
    for j = 1:n_ref
        fprintf('%14.4e', E_act_pt(b, j));
    end
    for j = 1:n_ref
        fprintf('%14.4e', E_nom_pt(b, j));
    end
    fprintf('%16.4e%16.4e\n', E_act_total(b), E_nom_total(b));
end

% ---- Save to disk ---------------------------------------------------
save('-mat', fullfile('figures', 'tracking_error_table.mat'), ...
    'TrackingErrorTable', 'TrackingErrorMatrix', 'col_names');

fid = fopen(fullfile('figures', 'tracking_error_table.csv'), 'w');
if fid > 0
    % header
    for c = 1:numel(col_names)
        if c > 1, fprintf(fid, ','); end
        fprintf(fid, '%s', col_names{c});
    end
    fprintf(fid, '\n');
    % rows
    for b = 1:N_batches
        for c = 1:size(TrackingErrorMatrix, 2)
            if c > 1, fprintf(fid, ','); end
            if c == 1
                fprintf(fid, '%d', TrackingErrorMatrix(b, c));
            else
                fprintf(fid, '%.6e', TrackingErrorMatrix(b, c));
            end
        end
        fprintf(fid, '\n');
    end
    fclose(fid);
    fprintf(['plot_results: tracking error table saved to ' ...
             'tracking_error_table.mat and tracking_error_table.csv\n']);
else
    warning('plot_results: could not open tracking_error_table.csv for writing.');
end

fprintf('plot_results: plotting completed.\n');
end


% =========================================================================
% Local helper: select_indices (kept for backward compatibility)
% =========================================================================
function idx = select_indices(N_total, k)
    if N_total <= 0
        idx = [];
    elseif N_total <= k
        idx = 1:N_total;
    else
        idx = unique(round(linspace(1, N_total, k)));
    end
end
