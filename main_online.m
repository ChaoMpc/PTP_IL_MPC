% =========================================================================
% main.m -- Crystallization batch DyPLS-IL-MPC simulation
% =========================================================================
clear; clc; close all;
if exist('OCTAVE_VERSION','builtin'), pkg load control optim; end

fprintf('=== Crystallization DyPLS-IL-MPC Simulation ===\n');

%% Step 0: 加载离线辨识结果
mat_filename = 'identified_model.mat';
if ~exist(mat_filename, 'file')
    error('未找到模型文件！请先运行 run_identification.m 进行系统辨识。');
end
fprintf('加载模型数据...\n');
load(mat_filename, 'Ident_data', 'Val_data', 'Model', ...
     'UncSets', 'Tube', 'CV_Data', 'Params');

% ★ 如果需要重设控制批次或随机种子，可以在这里覆盖
Params.N_batches = 20;
if isfield(Params,'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng','builtin'), rng(Params.rng_seed);
    else, rand('seed',Params.rng_seed); randn('seed',Params.rng_seed); end
end
if ~isfield(Params, 'enable_init_perturb')
    Params.enable_init_perturb = true;
end

% 1. 最优可达点计算
fprintf('计算最优可达点...\n');
y_opt_track = compute_optimal_reachable(Ident_data, Model, Tube, Params);

% ★ 新增：打印参考点与最优可达点对比
fprintf('\n=== Reference vs. Optimal Reachable Point ===\n');
fprintf('  Time (min) | True Ref Point | Optimal Reach. | Status\n');
fprintf(' ------------+----------------+----------------+----------------\n');
for j = 1:length(Params.Kp)
    k_idx = Params.Kp(j);

    % 计算真实的物理参考值 (基于基准轨迹 + 偏差)
    y_ref_val = Params.y_base(Params.track_idx, k_idx + 1) + Params.Ref_points(j);

    % 提取最优可达值
    y_opt_val = y_opt_track(j);

    % 判断是否可达 (允许 1e-4 的计算误差)
    if abs(y_opt_val - y_ref_val) > 1e-4
        status_str = 'NO (Unreachable)';
    else
        status_str = 'Yes (Reachable)';
    end

    % 打印该时刻的对比信息
    fprintf('    %3d      |    %.5f     |    %.5f     | %s\n', ...
            k_idx, y_ref_val, y_opt_val, status_str);
end
fprintf('===============================================================\n\n');


%% Step 2: 直接运行 Proposed 方法
fprintf('\n=== Running PROPOSED method ===\n');
Data_prop = run_ilmpc(Ident_data, Model, Tube, UncSets, Params);
Metrics_prop = compute_metrics(Data_prop, Params, Model, UncSets);

%% Step 3: 直接运行 Xue 方法
fprintf('\n=== Running XUE method ===\n');
Data_xue = run_xue_ilmpc(Ident_data, Model, Params);
Metrics_xue = compute_metrics(Data_xue, Params, Model, UncSets);

%% Step 4: Summary
fprintf('\n--- 约束违背与饱和详细对比 ---\n');
fprintf('Proposed 方法: 输入违背 %d 次, 输出违背 %d 次, 执行器饱和 %d 次\n', ...
    Metrics_prop.n_violations_v, Metrics_prop.n_violations_y, Metrics_prop.n_saturations);
fprintf('Xue 方法     : 输入违背 %d 次, 输出违背 %d 次, 执行器饱和 %d 次\n', ...
    Metrics_xue.n_violations_v, Metrics_xue.n_violations_y, Metrics_xue.n_saturations);

fprintf('\n===================== Detailed Comparison =====================\n');
fprintf('%-30s | %-15s | %-15s\n', 'Metric', 'Proposed (Tube)', 'Xue (ILC)');
fprintf('-------------------------------+-----------------+-----------------\n');
fprintf('%-30s | %-15d | %-15d\n', 'Input Violations', Metrics_prop.n_violations_v, Metrics_xue.n_violations_v);
fprintf('%-30s | %-15d | %-15d\n', 'Output Violations', Metrics_prop.n_violations_y, Metrics_xue.n_violations_y);
fprintf('%-30s | %-15d | %-15d\n', 'Actuator Saturations', Metrics_prop.n_saturations, Metrics_xue.n_saturations);
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Max Output Violation', Metrics_prop.max_violation_y, Metrics_xue.max_violation_y);
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Final Nominal Error', Data_prop.Error_norm_nominal(end), Data_xue.Error_norm_nominal(end));
fprintf('%-30s | %-15.4e | %-15.4e\n', 'Final Actual Error', Data_prop.Error_norm_actual(end), Data_xue.Error_norm_actual(end));
fprintf('%-30s | %-15.4f | %-15.4f\n', 'Mean QP Time (s)', mean(Data_prop.QP_times), mean(Data_xue.QP_times));
fprintf('===============================================================\n');

%% Step 5: Plots
if isfield(Params, 'plot_results') && Params.plot_results
    plot_results(Data_prop, Data_xue, Params, CV_Data, Val_data, ...
                 Metrics_prop, Metrics_xue, Tube, Model, y_opt_track);
end

%% Step 6: Data Persistence (数据保留)
save('simulation_results.mat', 'Data_prop', 'Data_xue', ...
     'Metrics_prop', 'Metrics_xue', 'Params', 'Model');

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


%% Step 8: Export Figures
try
    export_all_figures('figures');
catch ME
    fprintf('[export] failed: %s\n', ME.message);
end

fprintf('\n=== Simulation completed ===\n');
