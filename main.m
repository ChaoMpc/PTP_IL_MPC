% ========================================================================
% main.m -- DyPLS-IL-MPC 完整闭环仿真主程序
%
% 被控对象 : Wood-Berry 精馏塔 (2x2 过程)
%
% 针对审稿人意见的说明：
%   本脚本不仅包含了基础的 ILC-MPC 闭环运行，还额外集成了用于回应
%   审稿人质疑的扩展测试（包括交叉验证参数、自由仿真验证、不可达参考点
%   直接求解、闭环数据覆盖率统计、以及非相同初始状态测试）。
% ========================================================================

clear; clc; close all;

% 如果在 Octave 下运行，加载必要的控制与优化工具箱
if exist('OCTAVE_VERSION', 'builtin')
    pkg load control optim;
end

fprintf('=== DyPLS-IL-MPC Simulation (Wood-Berry) ===\n');

% ---------- Step 1: 参数初始化 ----------
% 调用 init_params.m 获取所有配置。
% 注意：请确保 init_params.m 中的 A_lv 和 n_arx 已经设置为了离线调参脚本输出的最优值
Params = init_params();

% 设定全局随机数种子，保证每次运行生成的噪声、PRBS 序列完全一致（便于复现结果）
if isfield(Params, 'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng', 'builtin'), rng(Params.rng_seed);
    else, rand('seed', Params.rng_seed); randn('seed', Params.rng_seed); end
end

% ---------- Step 2: 识别数据 ----------
% 生成用于模型辨识的 PRBS 激励数据
[Ident_data, Params] = gen_prbs_excitation(Params);
fprintf('main: identification data generated (N=%d).\n', size(Ident_data.X0,1));

% ---------- Step 3: 独立验证数据 ----------
% 生成独立验证数据（不同随机种子，未参与模型训练）
% 用于后续的交叉验证和 Free-run 仿真（回应审稿人 R1-Minor 14）
[Val_data, Params] = gen_validation_data(Params);
fprintf('main: validation data generated (N_val=%d).\n', size(Val_data.X0,1));

% ---------- Step 4: 建模 (使用 init_params 中设定的最优参数) ----------
% 使用辨识数据提取潜变量，并拟合 ARX 模型
Model = build_dypls(Ident_data, Params);
fprintf('main: DyPLS-ARX model built (A=%d, n_arx=%d).\n', Params.A_lv, Params.n_arx);

% ---------- Step 5: 独立验证数据上的自由仿真 ----------
% 回应审稿人 R1-Major 8：不再仅仅展示平凡的 UQ^T 重构，而是展示真实的动态预测能力
fprintf('\n--- Free-run validation ---\n');

% 1. 全局指标 (评估模型在整个长验证集上的整体发散程度)
[Y0_sim, nrmses, vafs] = free_run_validation(Model, Val_data, Params);
fprintf('  Global Validation NRMSE: [%s]\n', num2str(nrmses, '%.4f '));
fprintf('  Global Validation VAF  : [%s] %%\n', num2str(vafs, '%.4f '));

% 2. 局部对齐指标 (截取一段长度为 60 的窗口，从起点对齐，用于放进取稿论文中)
% 这样能向审稿人展示：给一个准确的起点，模型能在此后 60 步内多准地预测动态和延迟。
start_idx = 1; window_len = 60;
[Y0_sim_win, nrmses_win, vafs_win] = free_run_validation(Model, Val_data, Params, start_idx, window_len);

% 将局部验证结果打包，传递给 plot_results.m 用于绘制 Fig 4
ValResults.Y0_sim     = Y0_sim_win;
ValResults.nrmses     = nrmses_win;
ValResults.vafs       = vafs_win;
ValResults.Val_data   = Val_data;
ValResults.start_idx  = start_idx;
ValResults.window_len = window_len;

% ---------- Step 6: 不确定集估计 ----------
% 计算动态残差 eta 和静态重构残差 r 的边界（基于开环辨识数据）
UncSets = estimate_uncertainty(Model, Ident_data, Params);
fprintf('main: uncertainty sets estimated.\n');

% ---------- Step 7: Tube 参数与约束紧化 ----------
% 计算反馈增益 K、不变集椭球 P，以及输入输出的约束收紧量 delta_v, delta_y
Tube = compute_tube(Ident_data, Model, UncSets, Params);
fprintf('main: Tube-MPC parameters computed. Spectral radius = %.4f\n', Tube.spectral_radius);

% ---------- Step 7.5: 打印不确定集与不变集参数 (回应 R1-Major 4) ----------
fprintf('\n=== Uncertainty and Invariant Set Parameters ===\n');

% 1. 不确定集边界参数 (Uncertainty bounds)
fprintf('[Uncertainty Sets]\n');
fprintf('  eta_bound (ARX innovation): [%s]\n', num2str(UncSets.eta_bound(:)', '%.4e '));
fprintf('  r_bound   (PLS residual)  : [%s]\n', num2str(UncSets.r_bound(:)', '%.4e '));

% 2. 椭圆形状矩阵 W_eta 和 W_r 的对角元素 (等于边界平方)
fprintf('\n[Ellipsoid Shape Matrices (Diagonal Elements)]\n');
fprintf('  W_eta (diag) : [%s]\n', num2str(diag(Tube.W_eta)', '%.4e '));
fprintf('  W_r   (diag) : [%s]\n', num2str(diag(Tube.W_r)', '%.4e '));

% 3. 不变集椭球矩阵 P_e (Tube.P) 的特征值信息
% 由于 P_e 维数较高，这里输出其最小/最大特征值，以及条件数
eig_P = eig(Tube.P);
fprintf('\n[Invariant Ellipsoid Matrix P_e]\n');
fprintf('  Dimension of P_e  : %d x %d\n', size(Tube.P,1), size(Tube.P,2));
fprintf('  min(eig(P_e))     : %.4e\n', min(eig_P));
fprintf('  max(eig(P_e))     : %.4e\n', max(eig_P));
fprintf('  Condition Number  : %.4e\n', max(eig_P)/min(eig_P));
fprintf('  trace(P_e)        : %.4e\n', trace(Tube.P));

% 4. 约束紧化量 (Tightening margins)
fprintf('\n[Tightening Margins]\n');
fprintf('  delta_v : [%s]\n', num2str(Tube.delta_v(:)', '%.4e '));
fprintf('  delta_y : [%s]\n', num2str(Tube.delta_y(:)', '%.4e '));

% 5. 算法调节参数
fprintf('\n[Tuning Parameters]\n');
if isfield(Params, 'tube_alpha'), fprintf('  alpha (Tube) : %.4f\n', Params.tube_alpha); else, fprintf('  alpha (Tube) : 0.2 (default)\n'); end
fprintf('  beta  (Ellip): 1.0 (used in Minkowski sum bound)\n');
fprintf('================================================\n\n');

% (可选) 保存到文件，方便日后查用
if ~exist('figures','dir'), mkdir('figures'); end
fid = fopen(fullfile('figures', 'uncertainty_params.txt'), 'w');
fprintf(fid, 'eta_bound = %s\n', num2str(UncSets.eta_bound(:)', '%.6e '));
fprintf(fid, 'r_bound   = %s\n', num2str(UncSets.r_bound(:)', '%.6e '));
fprintf(fid, 'W_eta_diag = %s\n', num2str(diag(Tube.W_eta)', '%.6e '));
fprintf(fid, 'W_r_diag   = %s\n', num2str(diag(Tube.W_r)', '%.6e '));
fprintf(fid, 'P_e_eigs   = %s\n', num2str(eig_P', '%.6e '));
fprintf(fid, 'delta_v    = %s\n', num2str(Tube.delta_v(:)', '%.6e '));
fprintf(fid, 'delta_y    = %s\n', num2str(Tube.delta_y(:)', '%.6e '));
fclose(fid);

% ---------- Step 8: 不可达参考点直接求解验证 (Problem 55) ----------
% 回应审稿人 R1-Major 9：直接求解带约束的优化问题，计算出最优可达解 y*_s
% 用于在图上证明：实际轨迹虽然没到达参考点，但已经到达了物理上能到达的最好位置。
fprintf('\n--- Direct verification of unreachable reference (Problem 55) ---\n');
VerifyResult = verify_unreachability(Ident_data, Model, Tube, Params);
fprintf('  y*_s(k_i) at Kp = [%s]  (wt%% dev)\n', ...
    num2str(VerifyResult.y_s_at_Kp(:,1)', '%.4f '));
fprintf('  Reference     at Kp = [%s]  (wt%% dev)\n', ...
    num2str(Params.Ref_points(:)', '%.4f '));
fprintf('  Nominal error at y*_s : [%s]\n', ...
    num2str(VerifyResult.nom_error_at_Kp(:)', '%.4f '));
fprintf('  Active constraints: %d\n', numel(VerifyResult.active));

% ---------- Step 9: 主闭环仿真 (IL-MPC) ----------
% 运行 15 个批次的闭环控制（每批开始求解一次 QP，批内执行 Tube 反馈）
Data = run_ilmpc(Ident_data, Model, Tube, UncSets, Params);

% ---------- Step 10: 闭环指标汇总 ----------
% 回应审稿人 R1-Major 5 & 6：统计实际约束违例次数，以及不确定集的闭环覆盖率
metrics = compute_metrics(Data, Params, Model, UncSets);

fprintf('\n=== Closed-loop metrics ===\n');
fprintf('  Constraint violations (input): %d (max %+.2e)\n', ...
    metrics.n_violations_v, metrics.max_violation_v);
fprintf('  Constraint violations (output): %d (max %+.2e)\n', ...
    metrics.n_violations_y, metrics.max_violation_y);
fprintf('  eta coverage: %.2f%% (%d samples)\n', 100*metrics.coverage_eta, ...
    metrics.n_eta_samples);
fprintf('  r   coverage: %.2f%% (%d samples)\n', 100*metrics.coverage_r, ...
    metrics.n_r_samples);
fprintf('  Mean QP solve time: %.4f s\n', mean(metrics.qp_times));

% ---------- Step 11 (测试部分): 非相同初始状态测试 (R1-Major 10) ----------
% 为了证明算法对初始状态波动的鲁棒性，我们在相同噪声下跑两次：
% 第一次：初始状态无偏差（Nominal）
% 第二次：初始状态人为加入偏差（Perturbed）
if isfield(Params, 'run_init_state_test') && Params.run_init_state_test
    fprintf('\n=== Test: Non-identical initial states (x0 + Z_e,inf) ===\n');

    % 设定一个固定的测试种子，确保“标称”和“扰动”两次仿真的随机噪声序列完全一致
    seed_test = Params.rng_seed + 1000;
    if exist('rng', 'builtin'), rng(seed_test); else, randn('seed', seed_test); end

    % 1. 跑标称情况 (Nominal) - 无初始偏差
    Params_temp_nom = Params;
    Params_temp_nom.init_state_variation = 0.0;
    Data_nom = run_ilmpc(Ident_data, Model, Tube, UncSets, Params_temp_nom);
    metrics_nom = compute_metrics(Data_nom, Params, Model, UncSets);

    % 2. 跑初始偏差情况 (Perturbed) - 在 mRPI 边界内注入 80% 的偏差
    % 重置随机种子，保证与标称情况面对相同的噪声干扰
    if exist('rng', 'builtin'), rng(seed_test); else, randn('seed', seed_test); end
    Params_temp_pert = Params;
    Params_temp_pert.init_state_variation = 0.15;
    Data_pert = run_ilmpc(Ident_data, Model, Tube, UncSets, Params_temp_pert);
    metrics_pert = compute_metrics(Data_pert, Params, Model, UncSets);

    % 打印对比结果：如果两次的约束违反数都是 0，说明 Tube 成功吸收了初始偏差
    fprintf('  [Nominal Init]   Violations: v=%d, y=%d | Final Act Err=%.4e\n', ...
        metrics_nom.n_violations_v, metrics_nom.n_violations_y, Data_nom.Error_norm_actual(end));
    fprintf('  [Perturbed Init] Violations: v=%d, y=%d | Final Act Err=%.4e\n', ...
        metrics_pert.n_violations_v, metrics_pert.n_violations_y, Data_pert.Error_norm_actual(end));

    % 将两次测试的结果保存到 Data 结构中，供 plot_results 画出 Fig 8 对比图
    Data_init_test.Nominal   = Data_nom;
    Data_init_test.Perturbed = Data_pert;
    Data.init_test = Data_init_test;
end
% 在 Step 12: 绘图 之前，补充 MC 汇总和所有测试对比数据
fprintf('\n--- Summary for Validation Table ---\n');

% 1. 主仿真指标
summary_metrics.main = metrics;

% 2. 初始状态扰动指标 (如果跑了的话)
if isfield(Data, 'init_test')
    summary_metrics.pert = compute_metrics(Data.init_test.Perturbed, Params, Model, UncSets);
end

% 3. 如果有 MC 循环 (假设写在外面并保存了)，把它的平均值和最大值提取出来
% 如果没有跑 MC，这一步可以跳过，代码会自动容错。
if isfield(Params, 'run_mc_test') && Params.run_mc_test
    % 假设有一个包含 mc_metrics 的结构体
    % summary_metrics.mc = mc_metrics;
end


% ---------- Step 12: 绘图 ----------
% 调用绘图函数。传入所有的附加信息（不可达点 y*_s、诊断指标、验证集 free-run 数据）
if isfield(Params, 'plot_results') && Params.plot_results
    plot_results(Data, Params, Model, Tube, VerifyResult, metrics, ValResults, summary_metrics);
    fprintf('main: plotting completed.\n');
end

% ---------- Step 13: 导出图形 ----------
% 将生成的所有 Figure 导出到 figures/ 文件夹下（包含 EPS, PDF, PNG）
if isfield(Params, 'plot_results') && Params.plot_results
    try
        export_all_figures(Params, 'figures');
        fprintf('main: figures exported to figures/\n');
    catch ME
        fprintf(2, '[export_all_figures] failed: %s\n', ME.message);
    end
end

fprintf('\n=== Simulation completed ===\n');
