clear; clc; close all;
if exist('OCTAVE_VERSION','builtin'), pkg load control optim; end

fprintf('=== Crystallization DyPLS-IL-MPC Simulation ===\n');

Params = init_params();
if isfield(Params,'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng','builtin'), rng(Params.rng_seed);
    else, rand('seed',Params.rng_seed); randn('seed',Params.rng_seed); end
end

if ~isfield(Params, 'enable_init_perturb')
    Params.enable_init_perturb = true;
end

fprintf('=== Data Consistency Check ===\n');
fprintf('  ctrl_start: %d\n', Params.ctrl_start);
fprintf('  n_arx: %d\n', Params.n_arx);
fprintf('  ARX training starts at k=%d\n', Params.ctrl_start + Params.n_arx);
fprintf('  Samples per batch: %d\n', Params.Ble - Params.ctrl_start - Params.n_arx + 1);
fprintf('  Total training samples: %d\n', ...
    (Params.Ble - Params.ctrl_start - Params.n_arx + 1) * Params.N_train_batches);
fprintf('===============================\n\n');

[Ident_data, Params] = gen_cryst_excitation(Params);
fprintf('main: identification data N=%d.\n', size(Ident_data.X0,1));

%figure('Name','PRBS Excitation Diagnostics','Color','w');
% 循环到 n_in_opt
%for i = 1:Params.n_in_opt
%    subplot(2,3,i); hold on; grid on;
%    plot(1:Params.Ble, Ident_data.X_abs(1:Params.Ble, i), 'b-', 'LineWidth', 1.2);
%    plot(1:Params.Ble, Params.u_base(i,:), 'r--', 'LineWidth', 1.2);
%    xlabel('k'); ylabel(sprintf('u_%d',i));
%    title(sprintf('u%d: switch every %d steps', i, Params.prbs_periods(i)));
%    if i == 1, legend('X_{abs}', 'u_{base}'); end
%end

[Val_data, Params] = gen_cryst_validation_data(Params);
fprintf('main: validation data N=%d.\n', size(Val_data.X0,1));

Model = build_dypls(Ident_data, Params);
if isfield(Params, 'plot_model') && Params.plot_model
    report_pls_diagnostics(Model, Ident_data);
end
verify_pls_dimensionality(Ident_data, Model, Params);

fprintf('\n--- ARX Regression Condition ---\n');
for a = 1:Model.A_lv
    n_arx = Params.n_arx;
    Phi = [];
    for b = 1:Ident_data.N_batches
        idx = (b-1)*Params.Ble + (1:Params.Ble);
        u_b = Model.U(idx,a); t_b = Model.T(idx,a);
        for k = (n_arx+1):Params.Ble
            Phi = [Phi; u_b(k-1:-1:k-n_arx)', t_b(k-1:-1:k-n_arx)'];
        end
    end
    fprintf('  LV %d: cond(Phi''Phi) = %.3e\n', a, cond(Phi'*Phi));
end

% ==== Step 5: Free-run validation ====
eval_start = Params.ctrl_start + 1;             % 61 行 (对应 60 min)
eval_len   = Params.Ble - Params.ctrl_start;    % 60 行

nrmses_sum = zeros(1, Params.n_out);
vafs_sum   = zeros(1, Params.n_out);
n_bat      = Params.N_val_batches;

Y0_sim = []; % 用于画图（只取第一批即可）

for b = 1:n_bat
    s_b = (b-1) * Params.Ble + eval_start;
    l_b = eval_len;
    [Y0_sim_dummy, n_loc, v_loc, ~] = free_run_validation(Model, Val_data, Params, s_b, l_b);

    nrmses_sum = nrmses_sum + n_loc;
    vafs_sum   = vafs_sum + v_loc;

    if b == 1
        Y0_sim = Y0_sim_dummy;
    end
end

nrmses = nrmses_sum / n_bat;
vafs   = vafs_sum / n_bat;

CV_Data.Y0_sim     = Y0_sim;
CV_Data.nrmses     = nrmses;
CV_Data.vafs       = vafs;
CV_Data.start_idx  = eval_start;
CV_Data.window_len = eval_len;

fprintf('  [60~120 min Free-run] NRMSE = [%s]\n', num2str(nrmses,'%.4f '));
fprintf('  [60~120 min Free-run] VAF   = [%s] %%\n', num2str(vafs,'%.4f '));

UncSets = estimate_uncertainty(Model, Ident_data, Params);
Tube = compute_tube(Ident_data, Model, UncSets, Params);
fprintf('\n=== Uncertainty / Tube parameters ===\n');
fprintf('  eta_bound   : [%s]\n', num2str(UncSets.eta_bound','%.4e '));
fprintf('  r_bound     : [%s]\n', num2str(UncSets.r_bound','%.4e '));
fprintf('  delta_v: [%s]\n', num2str(Tube.delta_v','%.4e '));
fprintf('  delta_y: [%s]\n', num2str(Tube.delta_y','%.4e '));
fprintf('  spectral radius: %.6f\n', Tube.spectral_radius);

fprintf('\n=== Computing Optimal Reachable Reference Points ===\n');
y_opt_track = compute_optimal_reachable(Ident_data, Model, Tube, Params);
fprintf('  Time (min) | True Reference | Optimal Reachable | Reachable?\n');
for j = 1:length(Params.Kp)
    k = Params.Kp(j);
    y_ref = Params.y_base(Params.track_idx, k + 1) + Params.Ref_points(j);
    reach_str = 'Yes';
    if abs(y_opt_track(j) - y_ref) > 1e-4
        reach_str = 'NO (Violation)';
    end
    fprintf('    %3d      |   %.5f    |     %.5f     |   %s\n', k, y_ref, y_opt_track(j), reach_str);
end

fprintf('\n=== Running PROPOSED method ===\n');
Data_prop = run_ilmpc(Ident_data, Model, Tube, UncSets, Params);
Metrics_prop = compute_metrics(Data_prop, Params, Model, UncSets);

fprintf('\n=== Running XUE method ===\n');
Data_xue = run_xue_ilmpc(Ident_data, Model, Params);
Metrics_xue = compute_metrics(Data_xue, Params, Model, UncSets);

fprintf('\n--- 约束违背与饱和详细对比 ---\n');
fprintf('Proposed 方法: 输入违背 %d 次, 输出违背 %d 次, 执行器饱和 %d 次\n', ...
    Metrics_prop.n_violations_v, Metrics_prop.n_violations_y, Metrics_prop.n_saturations);
fprintf('Xue 方法     : 输入违背 %d 次, 输出违背 %d 次, 执行器饱和 %d 次\n', ...
    Metrics_xue.n_violations_v, Metrics_xue.n_violations_y, Metrics_xue.n_saturations);

if Metrics_xue.n_saturations > 0
    fprintf('  💡 观察：Xue 方法由于无批次内反馈，输入多次触及边界（饱和），导致输出失配越界。\n');
end

%% Step 10: Summary & Data Export
fprintf('\n===================== Detailed Comparison =====================\n');
fprintf('%-30s | %-15s | %-15s\n', 'Metric', 'Proposed (Tube)', 'Xue (ILC)');
fprintf('-------------------------------+-----------------+-----------------\n');

fprintf('%-30s | %-15d | %-15d\n', 'Input Violations', ...
    Metrics_prop.n_violations_v, Metrics_xue.n_violations_v);
fprintf('%-30s | %-15d | %-15d\n', 'Output Violations', ...
    Metrics_prop.n_violations_y, Metrics_xue.n_violations_y);
fprintf('%-30s | %-15d | %-15d\n', 'Actuator Saturations', ...
    Metrics_prop.n_saturations, Metrics_xue.n_saturations);
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Max Output Violation', ...
    Metrics_prop.max_violation_y, Metrics_xue.max_violation_y);
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Final Nominal Error', ...
    Data_prop.Error_norm_nominal(end), Data_xue.Error_norm_nominal(end));
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Final Actual Error', ...
    Data_prop.Error_norm_actual(end), Data_xue.Error_norm_actual(end));
fprintf('%-30s | %-15.4f | %-15.4f\n', 'Mean QP Time (s)', ...
    mean(Data_prop.QP_times), mean(Data_xue.QP_times));
fprintf('===============================================================\n');

%% Step 11: Plots
if isfield(Params, 'plot_results') && Params.plot_results
    plot_results(Data_prop, Data_xue, Params, CV_Data, Val_data, ...
                 Metrics_prop, Metrics_xue, Tube, Model, y_opt_track);
end

%% Step 12: Data Persistence (数据保留)
fprintf('\n=== Saving results to workspace files ===\n');
% 1. 保存所有关键变量到 .mat 文件（供下次加载或进一步分析）
save('simulation_results.mat', 'Data_prop', 'Data_xue', ...
     'Metrics_prop', 'Metrics_xue', 'Params', 'Model');

% 2. 将对比数据导出为 CSV (可直接用 Excel 打开)
fid = fopen('comparison_table.csv', 'w');
fprintf(fid, 'Metric,Proposed (Tube),Xue (ILC)\n');
fprintf(fid, 'Input Violations,%d,%d\n', Metrics_prop.n_violations_v, Metrics_xue.n_violations_v);
fprintf(fid, 'Output Violations,%d,%d\n', Metrics_prop.n_violations_y, Metrics_xue.n_violations_y);
fprintf(fid, 'Actuator Saturations,%d,%d\n', Metrics_prop.n_saturations, Metrics_xue.n_saturations);
fprintf(fid, 'Max Output Violation,%.4e,%.4e\n', Metrics_prop.max_violation_y, Metrics_xue.max_violation_y);
fprintf(fid, 'Final Nominal Error,%.4e,%.4e\n', Data_prop.Error_norm_nominal(end), Data_xue.Error_norm_nominal(end));
fprintf(fid, 'Final Actual Error,%.4e,%.4e\n', Data_prop.Error_norm_actual(end), Data_xue.Error_norm_actual(end));
fprintf(fid, 'Mean QP Time (s),%.4f,%.4f\n', mean(Data_prop.QP_times), mean(Data_xue.QP_times));
fclose(fid);
fprintf('Results saved to simulation_results.mat and comparison_table.csv\n');

%% Step 13: Additional Plot - Error Convergence (误差收敛曲线)
figure('Name','Fig 8: Error Convergence','Color','w','Position',[200 200 700 500]);
hold on; grid on;
plot(1:Params.N_batches, Data_prop.Error_norm_actual, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', 'b');
plot(1:Params.N_batches, Data_xue.Error_norm_actual,  'r-s', 'LineWidth', 1.5, 'MarkerFaceColor', 'r');
set(gca, 'YScale', 'log'); % 使用对数坐标更能看出收敛趋势
xlabel('Batch Number', 'FontSize', 11);
ylabel('Actual Error Norm (log scale)', 'FontSize', 11);
title('Actual Error Convergence Comparison', 'FontSize', 12);
legend('Proposed (Tube IL-MPC)', 'Xue Method (ILC)', 'Location', 'best', 'FontSize', 10);
xlim([1, Params.N_batches]);

%% Step 14: Export Figures
try
    export_all_figures('figures');
catch ME
    fprintf('[export] failed: %s\n', ME.message);
end

fprintf('\n=== Simulation completed ===\n');

try
    export_all_figures('figures');
catch ME
    fprintf('[export] failed: %s\n', ME.message);
end
fprintf('\n=== Simulation completed ===\n');
