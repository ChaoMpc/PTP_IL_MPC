function plot_results(Data, Params, varargin)
% =========================================================================
% plot_results -- DyPLS-IL-MPC 结果可视化核心函数。
%
% 依照最新要求修改：
%   - Fig 1 拆分为独立的 y1 和 y2 图
%   - Fig 2 拆分为独立的 v1 和 v2 图
%   - 删除 Fig 5 (Final Batch)
%   - Fig 8 拆分为独立的 y1 和 y2 图
% =========================================================================

%% 0. 解析输入参数
Model = []; Tube = []; VerifyResult = []; metrics = []; ValResults = []; summary_metrics = [];
if numel(varargin) >= 1 && ~isempty(varargin{1}), Model        = varargin{1}; end
if numel(varargin) >= 2 && ~isempty(varargin{2}), Tube         = varargin{2}; end
if numel(varargin) >= 3 && ~isempty(varargin{3}), VerifyResult = varargin{3}; end
if numel(varargin) >= 4 && ~isempty(varargin{4}), metrics      = varargin{4}; end
if numel(varargin) >= 5 && ~isempty(varargin{5}), ValResults   = varargin{5}; end
if numel(varargin) >= 6 && ~isempty(varargin{6}), summary_metrics = varargin{6}; end

%% 1. 基础数据展开与单位转换
N         = Params.Ble;
N_batches = Params.N_batches;
n_out     = Params.n_out;
n_in      = Params.n_in;
Ts        = Params.Ts;

if isfield(Data,'y_ss') && ~isempty(Data.y_ss), y_ss = Data.y_ss(:); else, y_ss = Params.y_ss(:); end
if isfield(Data,'u_ss') && ~isempty(Data.u_ss), u_ss = Data.u_ss(:); else, u_ss = Params.u_ss(:); end

Y_dev_hist         = Data.Y_hist;
V_dev_hist         = Data.V_hist;
Y_dev_nominal_hist = Data.Y_nominal_hist;
V_dev_nominal_hist = Data.V_nominal_hist;

Err_actual  = Data.Error_norm_actual(:);
Err_nominal = Data.Error_norm_nominal(:);

y_phys_max = Params.y_phys_max(:); y_phys_min = Params.y_phys_min(:);
u_phys_max = Params.u_phys_max(:); u_phys_min = Params.u_phys_min(:);

Kp             = Params.Kp(:);
y_ref_phys     = Params.y_ref_phys(:);

Y_hist_phys         = bsxfun(@plus, Y_dev_hist,         reshape(y_ss, [n_out,1,1]));
Y_nominal_hist_phys = bsxfun(@plus, Y_dev_nominal_hist, reshape(y_ss, [n_out,1,1]));
V_hist_phys         = bsxfun(@plus, V_dev_hist,         reshape(u_ss, [n_in,1,1]));
V_nominal_hist_phys = bsxfun(@plus, V_dev_nominal_hist, reshape(u_ss, [n_in,1,1]));

%% 2. 提取 Tube 紧化后的约束边界
y_tight_max = []; y_tight_min = []; u_tight_max = []; u_tight_min = [];
if ~isempty(Tube) && isfield(Tube,'h_y_tight') && ~isempty(Tube.h_y_tight)
    n_out_loc = n_out;
    y_tight_max = Tube.h_y_tight(1:n_out_loc)          + y_ss;
    y_tight_min = -Tube.h_y_tight(n_out_loc+1:end)     + y_ss;
end
if ~isempty(Tube) && isfield(Tube,'h_v_tight') && ~isempty(Tube.h_v_tight)
    n_in_loc = n_in;
    u_tight_max = Tube.h_v_tight(1:n_in_loc)           + u_ss;
    u_tight_min = -Tube.h_v_tight(n_in_loc+1:end)      + u_ss;
end

%% 3. 时间向量与绘图样式
t_y = (1:N)'     * Ts;
t_v = (1:N-1)'   * Ts;

colors = lines(6);
lw_nom = 1.0; lw_act = 1.5; lw_bnd = 0.8; lw_tight = 1.0;
ms_ref = 6; ms_star = 8; fs_leg = 8;

desired_batches = [1, 3, 5, 15];
batch_list = desired_batches(desired_batches <= N_batches);
if isempty(batch_list), batch_list = 1; end
n_show = numel(batch_list);

%% =====================================================================
% Figure 1 -- 输出跟踪 (y1)
% =====================================================================
figure('Name','Output Tracking y1','Color','w'); hold on; grid on;
i = 1; % y1
h_bnd_hi = plot(t_y, y_phys_max(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd);
plot(t_y, y_phys_min(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd, 'HandleVisibility','off');
h_tight_hi = [];
if ~isempty(y_tight_max)
    h_tight_hi = plot(t_y, y_tight_max(i)*ones(N,1), 'm:', 'LineWidth', lw_tight);
    plot(t_y, y_tight_min(i)*ones(N,1), 'm:', 'LineWidth', lw_tight, 'HandleVisibility','off');
end
h_batch = zeros(1, n_show);
for j = 1:n_show
    b   = batch_list(j);
    clr = colors(mod(j-1, size(colors,1))+1, :);
    plot(t_y, Y_nominal_hist_phys(i,:,b+1), '--', 'Color', clr, 'LineWidth', lw_nom, 'HandleVisibility','off');
    h_batch(j) = plot(t_y, Y_hist_phys(i,:,b), '-', 'Color', clr, 'LineWidth', lw_act);
end
h_ref = []; h_star = [];
if ~isempty(Kp)
    h_ref = plot(t_y(Kp), y_ref_phys, 'ro', 'MarkerSize', ms_ref, 'MarkerFaceColor','r');
    if ~isempty(VerifyResult) && isfield(VerifyResult,'y_s_at_Kp')
        y_s_phys = VerifyResult.y_s_at_Kp(:, 1) + y_ss(1);
        h_star = plot(t_y(Kp), y_s_phys, 'bo', 'MarkerSize', ms_star, 'MarkerFaceColor','none', 'MarkerEdgeColor','b', 'LineWidth', 1.5);
    end
end
xlabel('Sample k'); ylabel('y_1 (wt%)');
leg_handles = h_batch;
leg_labels  = arrayfun(@(bb) sprintf('Batch %d', bb), batch_list, 'UniformOutput', false);
if ~isempty(h_ref),  leg_handles(end+1) = h_ref;  leg_labels{end+1} = 'Reference'; end
if ~isempty(h_star), leg_handles(end+1) = h_star; leg_labels{end+1} = 'y_s^*'; end
leg_handles = [leg_handles, h_bnd_hi]; leg_labels = [leg_labels, 'Original Constraint'];
if ~isempty(h_tight_hi), leg_handles(end+1) = h_tight_hi; leg_labels{end+1} = 'Tightened Constraint'; end
legend(leg_handles, leg_labels, 'Location','southeast', 'FontSize',fs_leg);
xlim([1, N]); ylim([y_phys_min(i)-0.1, y_phys_max(i)+0.1]);

%% =====================================================================
% Figure 2 -- 输出跟踪 (y2)
% =====================================================================
figure('Name','Output Tracking y2','Color','w'); hold on; grid on;
i = 2; % y2
h_bnd_hi = plot(t_y, y_phys_max(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd);
plot(t_y, y_phys_min(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd, 'HandleVisibility','off');
h_tight_hi = [];
if ~isempty(y_tight_max)
    h_tight_hi = plot(t_y, y_tight_max(i)*ones(N,1), 'm:', 'LineWidth', lw_tight);
    plot(t_y, y_tight_min(i)*ones(N,1), 'm:', 'LineWidth', lw_tight, 'HandleVisibility','off');
end
h_batch = zeros(1, n_show);
for j = 1:n_show
    b   = batch_list(j);
    clr = colors(mod(j-1, size(colors,1))+1, :);
    plot(t_y, Y_nominal_hist_phys(i,:,b+1), '--', 'Color', clr, 'LineWidth', lw_nom, 'HandleVisibility','off');
    h_batch(j) = plot(t_y, Y_hist_phys(i,:,b), '-', 'Color', clr, 'LineWidth', lw_act);
end
xlabel('Sample k'); ylabel('y_2 (wt%)');
leg_handles = [h_batch, h_bnd_hi];
leg_labels  = [arrayfun(@(bb) sprintf('Batch %d', bb), batch_list, 'UniformOutput', false), {'Original Constraint'}];
if ~isempty(h_tight_hi), leg_handles(end+1) = h_tight_hi; leg_labels{end+1} = 'Tightened Constraint'; end
legend(leg_handles, leg_labels, 'Location','southeast', 'FontSize',fs_leg);
xlim([1, N]);

%% =====================================================================
% Figure 3 -- 控制输入 (v1)
% =====================================================================
figure('Name','Control Inputs v1','Color','w'); hold on; grid on;
i = 1; % v1
h_bnd_hi = plot(t_v, u_phys_max(i)*ones(N-1,1), 'k--','LineWidth',lw_bnd);
plot(t_v, u_phys_min(i)*ones(N-1,1), 'k--','LineWidth',lw_bnd, 'HandleVisibility','off');
h_tight_hi = [];
if ~isempty(u_tight_max)
    h_tight_hi = plot(t_v, u_tight_max(i)*ones(N-1,1),'m:','LineWidth',lw_tight);
    plot(t_v, u_tight_min(i)*ones(N-1,1),'m:','LineWidth',lw_tight, 'HandleVisibility','off');
end
h_batch = zeros(1, n_show);
for j = 1:n_show
    b   = batch_list(j);
    clr = colors(mod(j-1, size(colors,1))+1, :);
    plot(t_v, V_nominal_hist_phys(i,:,b+1), '--', 'Color', clr, 'LineWidth', lw_nom, 'HandleVisibility','off');
    h_batch(j) = plot(t_v, V_hist_phys(i,:,b), '-', 'Color', clr, 'LineWidth', lw_act);
end
xlabel('Sample k'); ylabel('v_1 = R (lb/min)');
leg_handles = [h_batch, h_bnd_hi];
leg_labels  = [arrayfun(@(bb) sprintf('Batch %d', bb), batch_list, 'UniformOutput', false), {'Original Constraint'}];
if ~isempty(h_tight_hi), leg_handles(end+1) = h_tight_hi; leg_labels{end+1} = 'Tightened Constraint'; end
legend(leg_handles, leg_labels, 'Location','southeast', 'FontSize',fs_leg);
xlim([1, N-1]);

%% =====================================================================
% Figure 4 -- 控制输入 (v2)
% =====================================================================
figure('Name','Control Inputs v2','Color','w'); hold on; grid on;
i = 2; % v2
h_bnd_hi = plot(t_v, u_phys_max(i)*ones(N-1,1), 'k--','LineWidth',lw_bnd);
plot(t_v, u_phys_min(i)*ones(N-1,1), 'k--','LineWidth',lw_bnd, 'HandleVisibility','off');
h_tight_hi = [];
if ~isempty(u_tight_max)
    h_tight_hi = plot(t_v, u_tight_max(i)*ones(N-1,1),'m:','LineWidth',lw_tight);
    plot(t_v, u_tight_min(i)*ones(N-1,1),'m:','LineWidth',lw_tight, 'HandleVisibility','off');
end
h_batch = zeros(1, n_show);
for j = 1:n_show
    b   = batch_list(j);
    clr = colors(mod(j-1, size(colors,1))+1, :);
    plot(t_v, V_nominal_hist_phys(i,:,b+1), '--', 'Color', clr, 'LineWidth', lw_nom, 'HandleVisibility','off');
    h_batch(j) = plot(t_v, V_hist_phys(i,:,b), '-', 'Color', clr, 'LineWidth', lw_act);
end
xlabel('Sample k'); ylabel('v_2 = S (lb/min)');
leg_handles = [h_batch, h_bnd_hi];
leg_labels  = [arrayfun(@(bb) sprintf('Batch %d', bb), batch_list, 'UniformOutput', false), {'Original Constraint'}];
if ~isempty(h_tight_hi), leg_handles(end+1) = h_tight_hi; leg_labels{end+1} = 'Tightened Constraint'; end
legend(leg_handles, leg_labels, 'Location','southeast', 'FontSize',fs_leg);
xlim([1, N-1]);

%% =====================================================================
% Figure 5 -- ILC 误差收敛图
% =====================================================================
figure('Name','ILC Convergence','Color','w'); hold on; grid on;
h1 = plot(1:N_batches, Err_actual,  'b-o','LineWidth',1.5,'MarkerSize',7, 'MarkerFaceColor','b');
h2 = plot(1:N_batches, Err_nominal, 'r-s','LineWidth',1.5,'MarkerSize',7, 'MarkerFaceColor','r');
xlim([1, N_batches]); xticks(1:N_batches);
xlabel('Batch number'); ylabel('Error norm at reference points');
legend([h1 h2], {'Actual error','Nominal error'}, 'Location','best');
fprintf('plot_results: ILC error norm summary\n');
fprintf('  batch 1        : nominal = %.4e, actual = %.4e\n', Err_nominal(1), Err_actual(1));
fprintf('  batch %-3d     : nominal = %.4e, actual = %.4e\n', N_batches, Err_nominal(end), Err_actual(end));

%% =====================================================================
% Figure 6 -- 独立验证集 Free-run Simulation
% =====================================================================
if ~isempty(ValResults) && isfield(ValResults, 'Y0_sim')
    Val_data  = ValResults.Val_data;
    Y0_sim    = ValResults.Y0_sim;
    nrmses    = ValResults.nrmses;
    vafs      = ValResults.vafs;
    start_idx = ValResults.start_idx;
    end_idx   = start_idx + ValResults.window_len - 1;
    t_window  = start_idx:end_idx;
    Y_true_phys = Val_data.Y0(t_window,:) .* (Val_data.Y_std') + Val_data.Y_mean' + Val_data.y_ss';
    Y_sim_phys  = Y0_sim(t_window,:)      .* (Val_data.Y_std') + Val_data.Y_mean' + Val_data.y_ss';
    t_plot = t_window - 1;
    figure('Name', 'Free-run Validation', 'Color', 'w');
    for i = 1:n_out
        subplot(n_out, 1, i); hold on; grid on;
        plot(t_plot, Y_true_phys(:,i), 'b-', 'LineWidth', 1.5);
        plot(t_plot, Y_sim_phys(:,i),  'r--', 'LineWidth', 1.5);
        xlabel('Sample k'); ylabel(sprintf('y_%d (wt%%)', i));
        title(sprintf('y_%d | NRMSE = %.4f, VAF = %.2f%%', i, nrmses(i), vafs(i)));
        legend('True (validation)', 'Free-run Simulation', 'Location', 'best');
        xlim([0, max(t_plot)]);
    end
end

%% =====================================================================
% Figure 7 -- 初始状态对比测试 (y1)
% =====================================================================
if isfield(Data, 'init_test')
    D_nom  = Data.init_test.Nominal;
    D_pert = Data.init_test.Perturbed;
    Y_pert_phys = bsxfun(@plus, D_pert.Y_hist, reshape(y_ss, [n_out,1,1]));

    figure('Name', 'Initial State Test y1', 'Color', 'w'); hold on; grid on;
    i = 1;
    plot(t_y, y_phys_max(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd);
    plot(t_y, y_phys_min(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd, 'HandleVisibility','off');
    if ~isempty(y_tight_max)
        plot(t_y, y_tight_max(i)*ones(N,1), 'm:', 'LineWidth', lw_tight);
        plot(t_y, y_tight_min(i)*ones(N,1), 'm:', 'LineWidth', lw_tight, 'HandleVisibility','off');
    end
    plot(t_y, Y_hist_phys(i,:,1), 'b-', 'LineWidth', 1.5);
    plot(t_y, Y_hist_phys(i,:,end), 'b:', 'LineWidth', 2.0);
    plot(t_y, Y_pert_phys(i,:,1), 'r-', 'LineWidth', 1.5);
    plot(t_y, Y_pert_phys(i,:,end), 'r:', 'LineWidth', 2.0);
    xlabel('Sample k'); ylabel('y_1 (wt%)'); xlim([1, N]);

    h1 = plot(nan, nan, 'b-', 'LineWidth', 1.5); h2 = plot(nan, nan, 'b:', 'LineWidth', 2.0);
    h3 = plot(nan, nan, 'r-', 'LineWidth', 1.5); h4 = plot(nan, nan, 'r:', 'LineWidth', 2.0);
    h5 = plot(nan, nan, 'k--', 'LineWidth', lw_bnd); h6 = plot(nan, nan, 'm:', 'LineWidth', lw_tight);
    legend([h1 h2 h3 h4 h5 h6], ...
           {'Ideal Init (Batch 1)', 'Ideal Init (Final Batch)', ...
            'Perturbed Init (Batch 1)', 'Perturbed Init (Final Batch)', ...
            'Original Constraint', 'Tightened Constraint'}, ...
           'Location', 'southeast', 'FontSize', fs_leg);
end

%% =====================================================================
% Figure 8 -- 初始状态对比测试 (y2)
% =====================================================================
if isfield(Data, 'init_test')
    D_nom  = Data.init_test.Nominal;
    D_pert = Data.init_test.Perturbed;
    Y_pert_phys = bsxfun(@plus, D_pert.Y_hist, reshape(y_ss, [n_out,1,1]));

    figure('Name', 'Initial State Test y2', 'Color', 'w'); hold on; grid on;
    i = 2;
    plot(t_y, y_phys_max(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd);
    plot(t_y, y_phys_min(i)*ones(N,1), 'k--', 'LineWidth', lw_bnd, 'HandleVisibility','off');
    if ~isempty(y_tight_max)
        plot(t_y, y_tight_max(i)*ones(N,1), 'm:', 'LineWidth', lw_tight);
        plot(t_y, y_tight_min(i)*ones(N,1), 'm:', 'LineWidth', lw_tight, 'HandleVisibility','off');
    end
    plot(t_y, Y_hist_phys(i,:,1), 'b-', 'LineWidth', 1.5);
    plot(t_y, Y_hist_phys(i,:,end), 'b:', 'LineWidth', 2.0);
    plot(t_y, Y_pert_phys(i,:,1), 'r-', 'LineWidth', 1.5);
    plot(t_y, Y_pert_phys(i,:,end), 'r:', 'LineWidth', 2.0);
    xlabel('Sample k'); ylabel('y_2 (wt%)'); xlim([1, N]);

    h1 = plot(nan, nan, 'b-', 'LineWidth', 1.5); h2 = plot(nan, nan, 'b:', 'LineWidth', 2.0);
    h3 = plot(nan, nan, 'r-', 'LineWidth', 1.5); h4 = plot(nan, nan, 'r:', 'LineWidth', 2.0);
    h5 = plot(nan, nan, 'k--', 'LineWidth', lw_bnd); h6 = plot(nan, nan, 'm:', 'LineWidth', lw_tight);
    legend([h1 h2 h3 h4 h5 h6], ...
           {'Ideal Init (Batch 1)', 'Ideal Init (Final Batch)', ...
            'Perturbed Init (Batch 1)', 'Perturbed Init (Final Batch)', ...
            'Original Constraint', 'Tightened Constraint'}, ...
           'Location', 'southeast', 'FontSize', fs_leg);
end

%% =====================================================================
% Table 1 -- Monte Carlo / Robustness Validation
% =====================================================================
if ~isempty(metrics)
    fprintf('\n\n========================================================================================\n');
    fprintf('  Table: Monte Carlo / Robustness Validation Summary\n');
    fprintf('========================================================================================\n');
    fprintf(' %-24s | %-10s | %-10s | %-10s | %-10s | %-10s\n', ...
            'Scenario', 'InViol', 'OutViol', 'Cov_Eta', 'Cov_R', 'QP Time');
    fprintf('----------------------------------------------------------------------------------------\n');

    fprintf(' %-24s | %-10d | %-10d | %-9.2f%% | %-9.2f%% | %-8.4fs\n', ...
        'Nominal Init (Main)', metrics.n_violations_v, metrics.n_violations_y, ...
        100*metrics.coverage_eta, 100*metrics.coverage_r, mean(metrics.qp_times));

    if isfield(Data, 'init_test') && isfield(summary_metrics, 'pert')
        m_pert = summary_metrics.pert;
        fprintf(' %-24s | %-10d | %-10d | %-9.2f%% | %-9.2f%% | %-8.4fs\n', ...
            'Perturbed Init', m_pert.n_violations_v, m_pert.n_violations_y, ...
            100*m_pert.coverage_eta, 100*m_pert.coverage_r, mean(m_pert.qp_times));
    end

    if isfield(summary_metrics, 'mc') && ~isempty(summary_metrics.mc)
        mc = summary_metrics.mc;
        fprintf(' %-24s | %-10d | %-10d | %-9.2f%% | %-9.2f%% | %-8.4fs\n', ...
            sprintf('MC (%d runs, Mean)', length(mc.cov_eta)), ...
            max(mc.max_viol_v), max(mc.max_viol_y), ...
            100*mean(mc.cov_eta), 100*mean(mc.cov_r), mean(metrics.qp_times));
    end

    fprintf('========================================================================================\n');
    fprintf('  * Note: InViol/OutViol denote the number of samples violating physical constraints.\n');
    fprintf('  * Cov_Eta/Cov_R denote the empirical coverage rate of the uncertainty sets.\n\n');

    if ~exist('figures','dir'), mkdir('figures'); end
    fid = fopen(fullfile('figures', 'validation_table.csv'), 'w');
    if fid > 0
        fprintf(fid, 'Scenario,InViol,OutViol,Cov_Eta(%%),Cov_R(%%),Mean_QP_Time(s)\n');
        fprintf(fid, 'Nominal Init,%d,%d,%.2f,%.2f,%.4f\n', ...
            metrics.n_violations_v, metrics.n_violations_y, 100*metrics.coverage_eta, 100*metrics.coverage_r, mean(metrics.qp_times));
        if isfield(Data, 'init_test') && isfield(summary_metrics, 'pert')
            fprintf(fid, 'Perturbed Init,%d,%d,%.2f,%.2f,%.4f\n', ...
                m_pert.n_violations_v, m_pert.n_violations_y, 100*m_pert.coverage_eta, 100*m_pert.coverage_r, mean(m_pert.qp_times));
        end
        fclose(fid);
        fprintf('  [Saved] Validation table exported to figures/validation_table.csv\n');
    end
end

%% =====================================================================
% Tube 控制台诊断信息
% =====================================================================
if ~isempty(Tube)
    fprintf('\nplot_results: Tube diagnostics\n');
    if isfield(Tube,'spectral_radius'), fprintf('  Closed-loop spectral radius : %.6f\n', Tube.spectral_radius); end
    if isfield(Tube,'delta_v'), fprintf('  Input  delta_v              : [%s]\n', num2str(Tube.delta_v(:)','%.3e ')); end
    if isfield(Tube,'delta_y'), fprintf('  Output delta_y              : [%s]\n', num2str(Tube.delta_y(:)','%.3e ')); end
end

%% =====================================================================
% 控制台输出跟踪误差表 (并保存为 .mat)
% =====================================================================
n_ref = numel(Params.Kp);
E_act_pt = zeros(N_batches, n_ref);
E_nom_pt = zeros(N_batches, n_ref);
for b = 1:N_batches
    for j = 1:n_ref
        k_j = Params.Kp(j);
        E_act_pt(b,j) = Y_dev_hist(1, k_j, b)           - Params.Ref_points(j);
        E_nom_pt(b,j) = Y_dev_nominal_hist(1, k_j, b+1) - Params.Ref_points(j);
    end
end
E_act_total = sqrt(sum(E_act_pt.^2, 2));
E_nom_total = sqrt(sum(E_nom_pt.^2, 2));

fprintf('\nplot_results: tracking error table (wt%%)\n');
fprintf('%6s', 'batch');
for j = 1:n_ref, fprintf('%12s', sprintf('act Kp%d', j)); end
for j = 1:n_ref, fprintf('%12s', sprintf('nom Kp%d', j)); end
fprintf('%14s%14s\n', 'act total', 'nom total');

for b = 1:N_batches
    fprintf('%6d', b);
    for j = 1:n_ref, fprintf('%12.4f', E_act_pt(b,j)); end
    for j = 1:n_ref, fprintf('%12.4f', E_nom_pt(b,j)); end
    fprintf('%14.4f%14.4f\n', E_act_total(b), E_nom_total(b));
end

if ~exist('figures','dir'), mkdir('figures'); end
save('-mat', fullfile('figures','tracking_error_table.mat'), 'E_act_pt', 'E_nom_pt', 'E_act_total', 'E_nom_total');
fprintf('plot_results: plotting completed.\n');
end
