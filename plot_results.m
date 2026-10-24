function plot_results(Data_prop, Data_xue, Params, CV_Data, Val_data, ...
                      Metrics_prop, Metrics_xue, Tube, Model, y_opt_track)
% =========================================================================
% plot_results -- 7 Comparison Figures
% =========================================================================
if nargin < 8 || isempty(Tube),  has_tube  = false; else, has_tube = true;  end
if nargin < 9 || isempty(Model), has_model = false; else, has_model = true; end
if nargin < 10, y_opt_track = []; end

N         = Params.Ble;
N_batches = Params.N_batches;
n_in_phys = Params.n_in;
n_in_opt  = Params.n_in_opt;
n_out     = Params.n_out;
trk       = Params.track_idx;
if isempty(trk) || trk < 1 || trk > n_out, trk = 0; end

y_base = Params.y_base;
t_y = (0:N)'   * Params.Ts;
t_v = (1:N)'   * Params.Ts;

all_batches = 1:N_batches;

in_names  = {'T_{j,sp} (K)','F_c (mL/s)','F_{anti} (mL/s)', ...
             'F_{sol} (mL/s)','N_{stir} (rpm)','F_{seed} (g/min)'};
out_names = {'T_1 (K)','T_2 (K)','T_3 (K)','C_{ATR}','C_{NIR}','C_{\rho}'};

if trk > 0
    y_ref_phys = Params.y_base(trk, Params.Kp + 1) + Params.Ref_points(:)';
    t_ref      = Params.Kp * Params.Ts;
else
    y_ref_phys = []; t_ref = [];
end

%% =======================================================================
% FIG 1 : PROPOSED vs XUE TRACKING FOR TRACK_IDX ONLY (BATCH 15, 16, 17, 20)
%% =======================================================================
if trk > 0
    figure('Name','Fig 1: Proposed vs Xue tracking (y_4)', 'Color','w', ...
           'Position',[80, 80, 1000, 600]);
    hold on; grid on;

    % --- 约束线 ---
    h_pmax = plot(t_y, Params.y_phys_max(trk)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
    h_pmin = plot(t_y, Params.y_phys_min(trk)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
    h_tmax = []; h_tmin = [];
    if has_tube
        y_hi = Params.y_phys_max(trk) - Tube.delta_y(trk);
        y_lo = Params.y_phys_min(trk) + Tube.delta_y(n_out + trk);
        h_tmax = plot(t_y, y_hi*ones(N+1,1), 'r-.', 'LineWidth', 1.2);
        h_tmin = plot(t_y, y_lo*ones(N+1,1), 'r-.', 'LineWidth', 1.2);
    end

    % --- 绘制曲线 (Batch 15, 16, 17, 20) ---
    batches_compare = [15, 16, 17, 20];
    batches_compare = batches_compare(batches_compare <= N_batches);

    styles = {'-', '--', '-.', ':'};
    h_prop = zeros(1, numel(batches_compare));
    h_xue  = zeros(1, numel(batches_compare));

    for jb = 1:numel(batches_compare)
        b = batches_compare(jb);
        st = styles{mod(jb-1, numel(styles)) + 1};

        Y_act_prop = squeeze(Data_prop.Y_hist(trk,:,b));
        Y_act_xue  = squeeze(Data_xue.Y_hist(trk,:,b));

        h_prop(jb) = plot(t_y, Y_act_prop, ['r', st], 'LineWidth', 1.5);
        h_xue(jb)  = plot(t_y, Y_act_xue,  ['k', st], 'LineWidth', 1.2);
    end

    % --- 参考点和最优可达点 ---
    h_ref = plot(t_ref, y_ref_phys, 'ro', 'MarkerSize', 8, 'MarkerFaceColor','r');
    h_opt = [];
    if ~isempty(y_opt_track)
        h_opt = plot(t_ref, y_opt_track, 'gp', 'MarkerSize', 10, 'MarkerFaceColor','g');
        for jj = 1:length(t_ref)
            plot([t_ref(jj), t_ref(jj)], [y_ref_phys(jj), y_opt_track(jj)], 'k:', 'LineWidth', 1.2);
        end
    end

    xlabel('Time (min)', 'FontSize', 11); ylabel(out_names{trk}, 'FontSize', 11);
    title(sprintf('Comparison: Tracking y_%d %s (Batches 15, 16, 17, 20)', trk, out_names{trk}), 'FontSize', 12);
    xlim([t_y(1), t_y(end)]);

    % --- 构建双列图例 ---
    lh = [h_pmax, h_pmin];
    ll = {'y_{max} (phys)', 'y_{min} (phys)'};
    if has_tube
        lh = [lh, h_tmax, h_tmin];
        ll = [ll, {'y_{max} (tight)', 'y_{min} (tight)'}];
    end

    for jb = 1:numel(batches_compare)
        lh = [lh, h_prop(jb)];
        ll = [ll, {sprintf('Prop Batch %d', batches_compare(jb))}];
    end
    for jb = 1:numel(batches_compare)
        lh = [lh, h_xue(jb)];
        ll = [ll, {sprintf('Xue Batch %d', batches_compare(jb))}];
    end

    lh = [lh, h_ref];
    ll = [ll, {'True Reference'}];
    if ~isempty(h_opt)
        lh = [lh, h_opt];
        ll = [ll, {'Optimal Reachable'}];
    end

    legend(lh, ll, 'Location', 'southwest', 'NumColumns', 2, 'FontSize', 8);
end

%% =======================================================================
% FIG 2 : PROPOSED ALL OUTPUTS (ALL BATCHES, HIGHLIGHT 15, 16, 17)
%% =======================================================================
figure('Name','Fig 2: Proposed all outputs', 'Color','w', ...
       'Position',[100, 80, 1400, 800]);
for j = 1:n_out
    subplot(2, 3, j); hold on; grid on;

    if j == trk
        plot(t_y, Params.y_phys_max(j)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
        plot(t_y, Params.y_phys_min(j)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
        if has_tube
            y_hi = Params.y_phys_max(j) - Tube.delta_y(j);
            y_lo = Params.y_phys_min(j) + Tube.delta_y(n_out + j);
            plot(t_y, y_hi*ones(N+1,1), 'r-.', 'LineWidth', 1.0);
            plot(t_y, y_lo*ones(N+1,1), 'r-.', 'LineWidth', 1.0);
        end
    end

    for b = all_batches
        Y_act = squeeze(Data_prop.Y_hist(j,:,b));
        if b == 15 || b == 16
            plot(t_y, Y_act, '-', 'Color', [1, 0, 0], 'LineWidth', 1.2); % 红
        elseif b == 17
            plot(t_y, Y_act, '-', 'Color', [0, 0, 1], 'LineWidth', 1.2); % 蓝
        else
            plot(t_y, Y_act, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        end
    end

    if j == trk && ~isempty(y_ref_phys)
        plot(t_ref, y_ref_phys, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r');
        if ~isempty(y_opt_track)
            plot(t_ref, y_opt_track, 'gp', 'MarkerSize', 8, 'MarkerFaceColor','g');
        end
    end

    xlabel('Time (min)', 'FontSize', 11); ylabel(out_names{j}, 'FontSize', 11);
    title(sprintf('y_%d  %s', j, out_names{j}), 'FontSize', 11);
    xlim([t_y(1), t_y(end)]);
end
try, sgtitle('Fig 2: Proposed method - all outputs (Batches 15,16 Red, 17 Blue)', 'FontWeight', 'bold', 'FontSize', 14); catch, end

%% =======================================================================
% FIG 3 : PROPOSED ALL INPUTS (ALL BATCHES, HIGHLIGHT 15, 16, 17)
%% =======================================================================
figure('Name','Fig 3: Proposed all inputs', 'Color','w', ...
       'Position',[120, 60, 1400, 800]);
for i = 1:n_in_phys
    subplot(2, 3, i); hold on; grid on;

    plot(t_v, Params.u_phys_max(i)*ones(N,1), 'k--', 'LineWidth', 0.8);
    plot(t_v, Params.u_phys_min(i)*ones(N,1), 'k--', 'LineWidth', 0.8);

    if i <= n_in_opt && has_tube
        u_hi = Params.u_phys_max(i) - Tube.delta_v(i);
        u_lo = Params.u_phys_min(i) + Tube.delta_v(n_in_opt + i);
        plot(t_v, u_hi*ones(N,1), 'r-.', 'LineWidth', 1.0);
        plot(t_v, u_lo*ones(N,1), 'r-.', 'LineWidth', 1.0);
    end

    for b = all_batches
        if i <= n_in_opt
            V_act = squeeze(Data_prop.V_hist(i,:,b));
            if b == 15 || b == 16
                plot(t_v, V_act, '-', 'Color', [1, 0, 0], 'LineWidth', 1.2);
            elseif b == 17
                plot(t_v, V_act, '-', 'Color', [0, 0, 1], 'LineWidth', 1.2);
            else
                plot(t_v, V_act, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
            end
        else
            % F_seed
            plot(t_v, Params.u_base(i, :), '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        end
    end

    xlabel('Time (min)', 'FontSize', 11); ylabel(in_names{i}, 'FontSize', 11);
    title(sprintf('u_%d  %s', i, in_names{i}), 'FontSize', 11);
    xlim([t_v(1), t_v(end)]);
end
try, sgtitle('Fig 3: Proposed method - all inputs (Batches 15,16 Red, 17 Blue)', 'FontWeight', 'bold', 'FontSize', 14); catch, end

%% =======================================================================
% FIG 4 : XUE ALL OUTPUTS (ALL BATCHES, HIGHLIGHT 15, 16, 17)
%% =======================================================================
figure('Name','Fig 4: Xue all outputs', 'Color','w', ...
       'Position',[160, 80, 1400, 800]);
for j = 1:n_out
    subplot(2, 3, j); hold on; grid on;

    if j == trk
        plot(t_y, Params.y_phys_max(j)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
        plot(t_y, Params.y_phys_min(j)*ones(N+1,1), 'k--', 'LineWidth', 0.8);
        if has_tube
            y_hi = Params.y_phys_max(j) - Tube.delta_y(j);
            y_lo = Params.y_phys_min(j) + Tube.delta_y(n_out + j);
            plot(t_y, y_hi*ones(N+1,1), 'r-.', 'LineWidth', 1.0);
            plot(t_y, y_lo*ones(N+1,1), 'r-.', 'LineWidth', 1.0);
        end
    end

    for b = all_batches
        Y_xue = squeeze(Data_xue.Y_hist(j,:,b));
        if b == 15 || b == 16
            plot(t_y, Y_xue, '-', 'Color', [1, 0, 0], 'LineWidth', 1.2);
        elseif b == 17
            plot(t_y, Y_xue, '-', 'Color', [0, 0, 1], 'LineWidth', 1.2);
        else
            plot(t_y, Y_xue, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        end
    end

    if j == trk && ~isempty(y_ref_phys)
        plot(t_ref, y_ref_phys, 'ro', 'MarkerSize', 6, 'MarkerFaceColor','r');
        if ~isempty(y_opt_track)
            plot(t_ref, y_opt_track, 'gp', 'MarkerSize', 8, 'MarkerFaceColor','g');
        end
    end

    xlabel('Time (min)', 'FontSize', 11); ylabel(out_names{j}, 'FontSize', 11);
    title(sprintf('y_%d  %s', j, out_names{j}), 'FontSize', 11);
    xlim([t_y(1), t_y(end)]);
end
try, sgtitle('Fig 4: Xue method - all outputs (Batches 15,16 Red, 17 Blue)', 'FontWeight', 'bold', 'FontSize', 14); catch, end

%% =======================================================================
% FIG 5 : XUE ALL INPUTS (ALL BATCHES, HIGHLIGHT 15, 16, 17)
%% =======================================================================
figure('Name','Fig 5: Xue all inputs', 'Color','w', ...
       'Position',[180, 60, 1400, 800]);
for i = 1:n_in_phys
    subplot(2, 3, i); hold on; grid on;

    plot(t_v, Params.u_phys_max(i)*ones(N,1), 'k--', 'LineWidth', 0.8);
    plot(t_v, Params.u_phys_min(i)*ones(N,1), 'k--', 'LineWidth', 0.8);

    for b = all_batches
        if i <= n_in_opt
            V_xue = squeeze(Data_xue.V_hist(i,:,b));
            if b == 15 || b == 16
                plot(t_v, V_xue, '-', 'Color', [1, 0, 0], 'LineWidth', 1.2);
            elseif b == 17
                plot(t_v, V_xue, '-', 'Color', [0, 0, 1], 'LineWidth', 1.2);
            else
                plot(t_v, V_xue, '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
            end
        else
            % F_seed 固定
            plot(t_v, Params.u_base(i, :), '-', 'Color', [0.7 0.7 0.7], 'LineWidth', 0.8);
        end
    end

    xlabel('Time (min)', 'FontSize', 11); ylabel(in_names{i}, 'FontSize', 11);
    title(sprintf('u_%d  %s (Xue)', i, in_names{i}), 'FontSize', 11);
    xlim([t_v(1), t_v(end)]);
end
try, sgtitle('Fig 5: Xue method - all inputs (Batches 15,16 Red, 17 Blue)', 'FontWeight', 'bold', 'FontSize', 14); catch, end

%% =======================================================================
% FIG 6 : FREE-RUN VALIDATION (0-120min plot, 60-120min eval)
%% =======================================================================
if ~isempty(CV_Data) && isfield(CV_Data,'Y0_sim') && has_model
    Y0_sim = CV_Data.Y0_sim;
    s_idx = CV_Data.start_idx;
    w_len = CV_Data.window_len;
    e_idx = s_idx + w_len - 1;

    Y_std  = Model.Y_std(:)';
    Y_mean = Model.Y_mean(:)';

    Y_base_seg = y_base(:, s_idx+1:e_idx+1)';
    Y_sim_phys_60_120  = Y0_sim(s_idx:e_idx, :) .* Y_std + Y_mean + Y_base_seg;

    t_seg_full = (0:N)' * Params.Ts;
    Y_true_phys_full = Val_data.Y_abs(1:N, :);

    Y_sim_phys_full = Y_true_phys_full;
    Y_sim_phys_full(s_idx:e_idx, :) = Y_sim_phys_60_120;

    figure('Name','Fig 6: Free-run validation','Color','w', ...
           'Position',[100, 80, 1400, 800]);
    for j = 1:n_out
        subplot(2, 3, j); hold on; grid on;

        plot(t_seg_full(1:N), Y_true_phys_full(:,j), 'k-',  'LineWidth', 1.5);
        plot(t_seg_full(1:N), Y_sim_phys_full(:,j),  'r--', 'LineWidth', 1.8);

        xline(Params.ctrl_start * Params.Ts, 'b-.', 'LineWidth', 1.5, 'HandleVisibility', 'off');

        if isfield(CV_Data,'nrmses')
            title(sprintf('%s   NRMSE=%.3f  VAF=%.1f%%', out_names{j}, ...
                          CV_Data.nrmses(j), CV_Data.vafs(j)), 'FontSize', 11);
        else
            title(out_names{j}, 'FontSize', 11);
        end
        xlabel('Time (min)', 'FontSize', 11); ylabel(out_names{j}, 'FontSize', 11);
        xlim([0, Params.Ble * Params.Ts]);

        if j == 1
            legend('Measured','Free-run (60-120)','Location','northeast','FontSize', 9);
        end
    end
    try
        sgtitle('Fig 6: Free-run validation (Evaluation on 60-120 min)', ...
                'FontWeight','bold', 'FontSize', 14);
    catch
    end
end

%% =======================================================================
% FIG 7 : ERROR CONVERGENCE
%% =======================================================================
figure('Name','Fig 7: Error Convergence','Color','w','Position',[200 200 700 500]);
hold on; grid on;
plot(1:Params.N_batches, Data_prop.Error_norm_actual, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
plot(1:Params.N_batches, Data_xue.Error_norm_actual,  'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
set(gca, 'YScale', 'log');
xlabel('Batch Number', 'FontSize', 11); ylabel('Actual Error Norm (log scale)', 'FontSize', 11);
title('Actual Error Convergence Comparison', 'FontSize', 12);
legend('Proposed (Tube IL-MPC)', 'Xue Method (ILC)', 'Location', 'southeast', 'FontSize', 10);
xlim([1, Params.N_batches]);

end
