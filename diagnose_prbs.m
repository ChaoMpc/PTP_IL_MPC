% =========================================================================
% diagnose_prbs.m -- 诊断 PRBS 激励下的纯物理轨迹偏差 (分阶段控制版)
% =========================================================================
clear; clc; close all;
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end

fprintf('=== 诊断 PRBS 激励下的轨迹偏差 (重点评估控制阶段) ===\n');
Params = init_params();
Ble = Params.Ble;
ctrl_start = Params.ctrl_start;

%% 1. 内联生成 PRBS 信号
X_prbs = zeros(Ble, Params.n_in);

for i = 1:Params.n_in
    seed = Params.rng_seed + i;
    period = Params.prbs_periods(i);
    amp_low = -Params.prbs_amp(i);
    amp_high = Params.prbs_amp(i);

    % 保存当前 RNG 状态
    if exist('OCTAVE_VERSION', 'builtin')
        old_state = rand("state");
        rand("state", seed);
    else
        old_rng = rng;
        rng(seed, 'twister');
    end

% 生成位流与变幅值因子
n_switches = ceil(Ble / period);
bits = randi([0, 1], n_switches, 1);
amps = 0.3 + 0.7 * rand(n_switches, 1); % ★ 变幅值因子：[0.3, 1.0]

% 拼接信号
u_temp = zeros(Ble, 1);
for k = 1:n_switches
    s_idx = (k-1)*period + 1;
    e_idx = min(k*period, Ble);
    if bits(k) == 0
        u_temp(s_idx:e_idx) = amp_low * amps(k);  % ★ 应用变幅值
    else
        u_temp(s_idx:e_idx) = amp_high * amps(k); % ★ 应用变幅值
    end
end
    X_prbs(:, i) = u_temp;
end

% ★ 核心修复：前 60 分钟纯基准，从 ctrl_start 之后才叠加 PRBS ★
X_prbs(1:ctrl_start, :) = 0;

%% 2. 仿真物理系统（绝对输入）
X_abs = Params.u_base' + X_prbs;
% 物理截断
for k = 1:Ble
    X_abs(k,:) = max(min(X_abs(k,:), Params.u_phys_max'), Params.u_phys_min');
end

x_plant = Params.x0_nom;
Y_abs = zeros(Ble, Params.n_out);
for k = 1:Ble
    [y_meas, x_plant] = cryst_plant(x_plant, X_abs(k,:)', Params, true);
    Y_abs(k,:) = y_meas';
end

%% 3. 计算纯物理偏差
U_dev = X_abs - Params.u_base';                  % n_batch x n_in
Y_dev = Y_abs - Params.y_base(:, 2:end)';        % n_batch x n_out

%% ★ 4. 打印统计信息（仅聚焦控制阶段 ctrl_start 之后） ★
idx_ctrl = ctrl_start:Ble;
U_dev_ctrl = U_dev(idx_ctrl, :);
Y_dev_ctrl = Y_dev(idx_ctrl, :);

fprintf('\n--- 物理偏差的标准差 (仅在 %d~%d 分钟控制段统计) ---\n', ctrl_start, Ble);
fprintf('  u1(Tj_sp): %.4f | u2(Fc): %.4f | u3(Fanti): %.4f | u4(Fsol): %.4f | u5(Nstir): %.4f | u6(Fseed): %.4f\n', ...
    std(U_dev_ctrl(:,1)), std(U_dev_ctrl(:,2)), std(U_dev_ctrl(:,3)), std(U_dev_ctrl(:,4)), std(U_dev_ctrl(:,5)), std(U_dev_ctrl(:,6)));

fprintf('  y1(T1)   : %.4e | y2(T2)   : %.4e | y3(T3)   : %.4e\n', std(Y_dev_ctrl(:,1)), std(Y_dev_ctrl(:,2)), std(Y_dev_ctrl(:,3)));
fprintf('  y4(C_ATR): %.4e | y5(C_NIR): %.4e | y6(C_rho): %.4e\n', std(Y_dev_ctrl(:,4)), std(Y_dev_ctrl(:,5)), std(Y_dev_ctrl(:,6)));

fprintf('\n--- 物理偏差的极值范围 (控制段) ---\n');
fprintf('  y4(C_ATR): min=%.5f, max=%.5f, range=%.5f\n', min(Y_dev_ctrl(:,4)), max(Y_dev_ctrl(:,4)), max(Y_dev_ctrl(:,4))-min(Y_dev_ctrl(:,4)));

% 检查控制段 u4(Fsol) 对 y4(C_ATR) 的相关系数
corr_val = corrcoef(U_dev_ctrl(:,4), Y_dev_ctrl(:,4));
fprintf('\n  [相关性诊断] u4(Fsol) 与 y4(C_ATR) 在控制段的相关系数: %.4f\n', corr_val(1,2));
if abs(corr_val(1,2)) < 0.1
    fprintf('  ⚠️ 警告：相关系数极低，说明输入的扰动对浓度几乎没有产生线性影响！\n');
else
    fprintf('  ✅ 相关性良好，模型能够学到输入到输出的动态关系。\n');
end

%% 5. 画图
time = 1:Ble;
figure('Name', 'PRBS 激励下的轨迹偏差诊断', 'Color', 'w', 'Position', [100 100 1200 800]);

% 图1: 降温轨迹 Tj_sp 及其偏差
subplot(2,2,1);
plot(time, Params.u_base(1,:), 'r--', 'LineWidth', 1.5); hold on;
plot(time, X_abs(:,1), 'b-', 'LineWidth', 0.8);
plot([ctrl_start, ctrl_start], [min(X_abs(:,1)), max(X_abs(:,1))], 'k-.', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('T_{j,sp}');
title('T_{j,sp}: 基准轨迹 (红) vs 实际输入 (蓝)');
legend('u_{base}', 'u_{abs}', 'Control Start', 'Location', 'best'); grid on;

% 图2: 反溶剂偏差 u3 与进料偏差 u4
subplot(2,2,2);
plot(time, U_dev(:,3), 'r-', 'LineWidth', 1.2); hold on;
plot(time, U_dev(:,4), 'b-', 'LineWidth', 1.2);
plot([ctrl_start, ctrl_start], [min([U_dev(:,3);U_dev(:,4)]), max([U_dev(:,3);U_dev(:,4)])], 'k-.', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('Input Deviation');
title('PRBS 纯偏差输入 (u3, u4)');
legend('F_{anti} dev', 'F_{sol} dev', 'Control Start', 'Location', 'best'); grid on;

% 图3: 浓度基准与绝对输出
subplot(2,2,3);
plot(time, Params.y_base(4,2:end), 'r--', 'LineWidth', 1.5); hold on;
plot(time, Y_abs(:,4), 'b-', 'LineWidth', 0.8);
plot([ctrl_start, ctrl_start], [min(Y_abs(:,4)), max(Y_abs(:,4))], 'k-.', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('C_{ATR}');
title('C_{ATR}: 基准轨迹 (红) vs 实际测量 (蓝)');
legend('y_{base}', 'y_{abs}', 'Control Start', 'Location', 'best'); grid on;

% 图4: 浓度纯物理偏差 y4_dev
subplot(2,2,4);
plot(time, Y_dev(:,4), 'k-', 'LineWidth', 1.2); hold on;
plot([1 Ble], [0 0], 'r-', 'LineWidth', 1.0);
plot([ctrl_start, ctrl_start], [min(Y_dev(:,4)), max(Y_dev(:,4))], 'k-.', 'LineWidth', 2);
xlabel('Time (min)'); ylabel('C_{ATR} Deviation');
title(sprintf('纯浓度偏差 (控制段 Std = %.2e)', std(Y_dev_ctrl(:,4))));
legend('y_4 dev', 'Zero', 'Control Start', 'Location', 'best'); grid on;

fprintf('\n--- 诊断图已生成，请观察图 4 中 60 分钟之后的波动形态。 ---\n');
fprintf('如果 60 分钟后呈现对应的方波响应，且相关性高，说明分阶段模型能够完美工作。\n');
