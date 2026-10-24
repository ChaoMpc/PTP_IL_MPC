function Params = init_params()
if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end

%% 1. Random seeds
Params.rng_seed     = 2024;
Params.rng_seed_val = 2025;

%% 2. Time / batch parameters
Params.Ts      = 1.0; Params.Ts_int = 0.005; Params.Ble = 120;
Params.N_train_batches = 300; Params.N_val_batches = 5; Params.N_batches = 20;
Params.N = Params.N_train_batches * Params.Ble;
Params.N_val = Params.N_val_batches * Params.Ble;
Params.n_in  = 6; Params.n_out = 6;
Params.ctrl_start = 60;

% ★ 新增：控制与辨识的输入维度（剔除最后一维不起作用的晶种）
Params.n_in_opt = 5;

%% 4. DyPLS-ARX parameters
Params.A_lv       = 4;
Params.n_arx      = 6;
Params.arx_lambda = 1e-6;
Params.A_lv_cv_range  = [2, 3, 4];
Params.n_arx_cv_range = [4, 6, 8, 10];

%% 5. 绝对物理约束
Params.u_phys_min = [250.0; 10.0; 0.0; 0.0; 50.0; 0.0];
Params.u_phys_max = [330.0; 100.0; 10.0; 10.0; 500.0; 2.0];
Params.y_phys_min = [-1e6; -1e6; -1e6; 0.01; -1e6; -1e6];
Params.y_phys_max = [ 1e6;  1e6;  1e6; 0.04;  1e6;  1e6];

% ★ 修改：H_v 维数基于 n_in_opt (10x5)
Params.H_v = [ eye(Params.n_in_opt); -eye(Params.n_in_opt) ];
Params.H_y = [ eye(Params.n_out); -eye(Params.n_out) ];

%% 6. 物理与传感器参数
Params.rho_c = 1296; Params.k_v = 0.24; Params.b = 6.2; Params.g = 1.5;
Params.ln_kb = 35.0; Params.ln_kg = -2.5; Params.m_w = 0.45;
Params.C_pw = 4.18; Params.DHc = 20e3; Params.UA0 = 0.12 * 60; Params.tau_j = 2.0;
Params.F_c0 = 1.0; Params.N0 = 400; Params.nu_F = 0.5; Params.nu_N = 0.2;
Params.V_sol = 0.5e-3; Params.C_feed = 0.03; Params.k_anti = 0.8; Params.k_seed = 5e-4;
Params.noise_var = 0.01;

%% ★ 7. 真实工艺基准轨迹构建
T0 = 305.0; C0 = 0.0256;
Params.x0_nom = [T0; T0; C0; 1e-4; 1e-3; 1e-2; 1e-3];

Params.u_base = zeros(Params.n_in, Params.Ble);
t = (0:Params.Ble-1)';
t_norm = t / (Params.Ble-1);

Params.u_base(1, :) = 305 - 10 * (t_norm.^2);
Params.u_base(2, :) = 60.0;
Params.u_base(3, :) = 4.0 + 1.0 * (t_norm.^1.5);
Params.u_base(4, :) = 5.0;
Params.u_base(5, :) = 300.0;
Params.u_base(6, :) = 0.0;
Params.u_base(6, 1:5) = 1.0;

x_sim = Params.x0_nom;
Params.y_base = zeros(Params.n_out, Params.Ble + 1);
[y0, x_sim] = cryst_plant(x_sim, [], Params, false);
Params.y_base(:, 1) = y0;
for k = 1:Params.Ble
    [y, x_sim] = cryst_plant(x_sim, Params.u_base(:, k), Params, false);
    Params.y_base(:, k+1) = y;
end

%% ★ 8. 分阶段控制
Params.track_idx = 4;
Params.ctrl_start = 60;
Params.Kp = [90, 120];
%Params.y_ref_phys = [0.022; 0.018];
Params.y_ref_phys = [0.022; 0.015];
Params.Ref_points = Params.y_ref_phys - Params.y_base(4, Params.Kp + 1)';

Params.Q_weight = 100 * eye(Params.n_out);
Params.R_weight = 1e-3;
Params.P_horizon = Params.Ble - Params.ctrl_start + 1;
Params.M_horizon = Params.Ble;
Params.delta_t_max = 1;
Params.ddt_max = 3;

%% 9. PRBS
% ★ 修改：只针对前 5 个输入
Params.prbs_periods = [5; 5; 8; 8; 5; 8];
u_half = 0.5 * (Params.u_phys_max - Params.u_phys_min);
Params.prbs_amp = 0.2 * u_half;

Params.W_y = [1.0; 1.0; 1.0; 1.0; 1.0; 1.0];
Params.enable_init_perturb = true;

%% 其他设置
Params.safety_factor = 1.2; Params.q_level = 0.95;
Params.Q_lqr = 50; Params.R_lqr = 1; % 降低防爆
Params.plot_results = true;

% ★ 只提取前 5 列给 PLS 训练
Params.u_base_rep = Params.u_base(1:Params.n_in_opt, :)';
Params.y_base_rep = Params.y_base(:, 2:end)';
fprintf('init_params: 分阶段控制模式启动，剔除 F_seed，辨识维度 n_in_opt=5。\n');
end
