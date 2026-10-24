% test_xue.m
clear; clc;
if exist('OCTAVE_VERSION', 'builtin'), pkg load control optim; end

% 1. 生成假参数
Params.A_lv = 4; Params.n_arx = 12;
Params.n_in = 6; Params.n_out = 6; Params.Ble = 100;
Params.N_batches = 3;

% 假模型
n_state = 2*Params.n_arx - 1;
nx = Params.A_lv * n_state;
barA = 0.9*eye(nx); barB = 0.1*ones(nx, Params.A_lv);
barC = zeros(Params.A_lv, nx); for i=1:Params.A_lv, barC(i, (i-1)*n_state+1) = 1; end

Model.A_aug = [barA, barB; zeros(Params.A_lv, nx), eye(Params.A_lv)];
Model.B_aug = [barB; eye(Params.A_lv)];
Model.C_aug1 = [barC, zeros(Params.A_lv, Params.A_lv)];
Model.C_aug2 = [zeros(Params.A_lv, nx), eye(Params.A_lv)];

% 2. 计算 Kalman 增益 (修复不可观测模态)
nx_aug = size(Model.A_aug, 1);
Q_proc = 1e-5 * eye(nx_aug);
R_meas = 1e-4 * eye(Params.A_lv);

% 【核心修复】：给 A_aug 极小衰减，避开 dare 的极点报错
A_for_dare = Model.A_aug - 1e-5 * eye(nx_aug);

% 【核心修复】：接收第一个输出（Riccati 解），而不是第二个
[P_inf, ~, ~] = dare(A_for_dare', Model.C_aug1', Q_proc, R_meas);
K_kf = (P_inf * Model.C_aug1') / (Model.C_aug1 * P_inf * Model.C_aug1' + R_meas);

fprintf('P_inf 维度: %dx%d\n', size(P_inf,1), size(P_inf,2));
fprintf('K_kf 维度: %dx%d\n', size(K_kf,1), size(K_kf,2));

% 模拟 Kalman 更新
dx_minus = zeros(nx_aug, 1);
u_current_k = zeros(Params.A_lv, 1);
u_prev_k = zeros(Params.A_lv, 1);

innov = u_current_k - u_prev_k - Model.C_aug1 * dx_minus;
fprintf('innov 维度: %dx%d\n', size(innov,1), size(innov,2));

dx_hat = dx_minus + K_kf * innov;
fprintf('dx_hat 维度: %dx%d\n', size(dx_hat,1), size(dx_hat,2));
fprintf('测试通过！\n');
