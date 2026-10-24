% =========================================================================
% cryst_open_loop_analysis.m
% 批次结晶开环测试脚本 (基准降温 + 6通道随机阶跃激励)
%
% 目的：
%   1. 观察系统在真实宽幅操作下的非线性动态。
%   2. 确认 6 个输入通道的灵敏度（防止 PLS 降维时因某通道全零而报错）。
%   3. 为设计物理约束（y_phys_min/max）和参考点（Ref_points）提供真实依据。
% =========================================================================

clear; clc; close all;
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end

fprintf('=== Batch Crystallization Open-Loop Analysis (6-Input Step) ===\n');

% 1. 加载参数 (请确保调用的是最新的 init_params)
Params = init_params();

% 2. 设定开环输入轨迹
rng(42); % 固定随机种子以便复现
u_profile = zeros(Params.n_in, Params.Ble);

% --- (a) Tj_sp: 基准降温 + 随机阶跃 ---
% 设定基准降温轨迹: 从 310 K 缓慢降至 290 K
Tj_base = 310 - 20 * ((1:Params.Ble) / Params.Ble);

step_len = 10; % 每 10 步跳变一次
n_steps = ceil(Params.Ble / step_len);

Tj_steps = 2 * (2*rand(1, n_steps) - 1); % +/- 2 K 的随机波动
Tj_noise = kron(Tj_steps, ones(1, step_len));
Tj_noise = Tj_noise(1:Params.Ble);
u_profile(1, :) = Tj_base + Tj_noise;

% --- (b) 其余 5 个通道的随机阶跃 ---
% Fc: 冷却流量在 [20, 100] 之间随机
Fc_steps = 20 + 80 * rand(1, n_steps);
Fc_noise = kron(Fc_steps, ones(1, step_len));
u_profile(2, :) = Fc_noise(1:Params.Ble);

% Fanti: 反溶剂在 [0, 2] 之间随机
Fanti_steps = 2 * rand(1, n_steps);
u_profile(3, :) = kron(Fanti_steps, ones(1, step_len));

% Fsol: 溶液进料在 [0, 2] 之间随机
Fsol_steps = 2 * rand(1, n_steps);
u_profile(4, :) = kron(Fsol_steps, ones(1, step_len));

% Nstir: 搅拌在 [200, 450] 之间随机
Nstir_steps = 200 + 250 * rand(1, n_steps);
u_profile(5, :) = kron(Nstir_steps, ones(1, step_len));

% Fseed: 晶种在 [0, 0.5] 之间随机
Fseed_steps = 0.5 * rand(1, n_steps);
u_profile(6, :) = kron(Fseed_steps, ones(1, step_len));

% 应用物理约束限幅，防止越界
for k = 1:Params.Ble
    u_profile(:, k) = max(min(u_profile(:, k), Params.u_phys_max), Params.u_phys_min);
end

% 3. 初始化系统与存储矩阵
x_plant = Params.x0_nom;
Y_hist = zeros(Params.n_out, Params.Ble);
X_hist = zeros(7, Params.Ble); % 现在的状态是 7 维

% 4. 运行开环仿真
for k = 1:Params.Ble
    v_k = u_profile(:, k);
    % 注意：最新模型不需要 do_reset 参数，直接传入 is_noisy = true
    [y_meas, x_next] = cryst_plant(x_plant, v_k, Params, true);

    Y_hist(:, k) = y_meas;
    X_hist(:, k) = x_plant;

    x_plant = x_next;
end

% 5. 结果绘图 (重点展示后几个关键输出通道)
time = 1:Params.Ble;

figure('Name', 'Open-Loop Random Step Analysis', 'Color', 'w', 'Position', [100 100 1200 800]);

% --- 图 1: 浓度演化 ---
subplot(2,2,1);
plot(time, Y_hist(4, :), 'b-', 'LineWidth', 2); hold on; grid on;
plot(time, Y_hist(5, :), 'r--', 'LineWidth', 1.5);
plot(time, Y_hist(6, :), 'g:', 'LineWidth', 1.5);
xlabel('Time (min)'); ylabel('Concentration (kg/kg)');
title('Concentration Sensors (C_{ATR}, C_{NIR}, C_\rho)');
legend('C_{ATR}', 'C_{NIR}', 'C_\rho', 'Location', 'best');

% --- 图 2: FBRM 晶体尺寸 ---
subplot(2,2,2);
plot(time, Y_hist(7, :), 'm-', 'LineWidth', 2); grid on;
xlabel('Time (min)'); ylabel('FBRM Mean Size (\mum)');
title('Crystal Mean Size (L_{FBRM})');

% --- 图 3: 浊度 ---
subplot(2,2,3);
plot(time, Y_hist(8, :), 'c-', 'LineWidth', 2); grid on;
xlabel('Time (min)'); ylabel('Turbidity (S_{turb})');
title('Turbidity');

% --- 图 4: 随机阶跃输入 (以 Tj_sp 为例) ---
subplot(2,2,4);
plot(time, u_profile(1, :), 'r-', 'LineWidth', 1.5); hold on; grid on;
plot(time, u_profile(3, :), 'b--', 'LineWidth', 1.5);
xlabel('Time (min)'); ylabel('Input Value');
title('Input Profile: T_{j,sp} & F_{anti}');
legend('T_{j,sp} (K)', 'F_{anti} (mL/s)', 'Location', 'best');

% 6. 控制台输出诊断信息（极度关键，直接指导约束设定）
fprintf('\n--- Open-Loop Random Step Diagnostics ---\n');

fprintf('\n[1] Concentration Output Range:\n');
fprintf('    C_ATR (y4): min = %.4f, max = %.4f\n', min(Y_hist(4, :)), max(Y_hist(4, :)));
fprintf('    C_NIR (y5): min = %.4f, max = %.4f\n', min(Y_hist(5, :)), max(Y_hist(5, :)));
fprintf('    C_rho (y6): min = %.4f, max = %.4f\n', min(Y_hist(6, :)), max(Y_hist(6, :)));

fprintf('\n[2] Crystal & Turbidity Range:\n');
fprintf('    FBRM (y7) : min = %.4f, max = %.4f\n', min(Y_hist(7, :)), max(Y_hist(7, :)));
fprintf('    Turbidity (y8): min = %.4f, max = %.4f\n', min(Y_hist(8, :)), max(Y_hist(8, :)));

fprintf('\n[3] Suggested Tracking Reference (C_ATR):\n');
fprintf('    At t=40  : C_ATR = %.4f\n', Y_hist(4, 40));
fprintf('    At t=80  : C_ATR = %.4f\n', Y_hist(4, 80));
fprintf('    At t=120 : C_ATR = %.4f\n', Y_hist(4, 120));

fprintf('\n[4] Suggested Output Constraints (Heuristic):\n');
% 仅基于浓度通道，建议下限不低于 0.002，上限放大 10%
c_min = min([Y_hist(4, :), Y_hist(5, :), Y_hist(6, :)]);
c_max = max([Y_hist(4, :), Y_hist(5, :), Y_hist(6, :)]);
fprintf('    y_phys_min for C (y4-y6) -> safe to set >= %.4f\n', max(0.001, c_min * 0.9));
fprintf('    y_phys_max for C (y4-y6) -> safe to set >= %.4f\n', c_max * 1.1);

fprintf('\n=== Copy-Paste config for init_params.m ===\n');
fprintf('Params.y_ref_phys = [%.4f, %.4f, %.4f];\n', Y_hist(4, 40)*0.95, Y_hist(4, 80)*0.95, Y_hist(4, 120)*0.95);
