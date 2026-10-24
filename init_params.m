function Params = init_params()
% =========================================================================
% init_params -- Wood-Berry DyPLS-IL-MPC 参数
% 依照审稿人意见修改：
%   (1) n_arx=10 以处理 7 min 纯延迟 (审稿人 R1-Major 8)
%   (2) 独立验证数据设置 (R1-Minor 14)
%   (3) 初始状态变化、MC 噪声、失配测试开关 (R1-Major 6,7,8)
%   (4) LQR 权重选择理由 (R1-Minor 18)
% =========================================================================

if exist('OCTAVE_VERSION', 'builtin'), pkg load control; end

%% 1. 连续时间被控对象
[numP_1, denP_1] = pade2(1);
[numP_3, denP_3] = pade2(3);
[numP_7, denP_7] = pade2(7);

G11 = tf( 12.8 * numP_1, conv([16.7 1], denP_1));
G12 = tf(-18.9 * numP_3, conv([21.0 1], denP_3));
G21 = tf(  6.6 * numP_7, conv([10.9 1], denP_7));
G22 = tf(-19.4 * numP_3, conv([14.4 1], denP_3));
G   = [G11, G12; G21, G22];

Gd = [tf(0.5, [10 1]); tf(0.5, [10 1])];

%% 2. 离散化
Params.Ts      = 1;
Params.sys_d   = c2d(G,  Params.Ts);
Params.sys_d_d = c2d(Gd, Params.Ts);

%% 3. 数据生成
Params.N             = 4000;        % 识别数据长度
Params.N_val         = 2000;        % 独立验证数据长度 (R1-Minor 14)
Params.rng_seed_val  = 99;          % 验证数据独立种子
Params.Ble           = 60;
Params.N_batches     = 15;
Params.verbose_ilmpc = 1;

%% 4. 系统维度
Params.n_in  = 2;
Params.n_out = 2;

%% 5. DyPLS 参数
% n_arx = 10 覆盖 7 采样延迟 (审稿人 R1-Major 8)
Params.A_lv  = 2;
Params.n_arx = 10;
Params.arx_lambda = 1e-6;

% 交叉验证搜索范围（供 build_dypls 参考）
Params.A_lv_cv_range  = [1, 2, 3, 4, 5];
Params.n_arx_cv_range = [4, 6, 8, 10, 12];

%% 6. 稳态工作点
Params.u_ss = [1.95; 1.71];
Params.y_ss = [96.25; 0.50];

%% 7. 物理约束 (绝对→偏差，单点转换)
Params.u_phys_min = [1.8525; 1.6245];
Params.u_phys_max = [2.0475; 1.7955];
Params.y_phys_min = [95.0; 0.00];
Params.y_phys_max = [97.5; 1.00];

Params.u_dev_min = Params.u_phys_min - Params.u_ss;
Params.u_dev_max = Params.u_phys_max - Params.u_ss;
Params.y_dev_min = Params.y_phys_min - Params.y_ss;
Params.y_dev_max = Params.y_phys_max - Params.y_ss;

Params.H_v = [ eye(Params.n_in); -eye(Params.n_in) ];
Params.H_y = [ eye(Params.n_out); -eye(Params.n_out) ];
Params.h_v_phys = [Params.u_dev_max; -Params.u_dev_min];
Params.h_y_phys = [Params.y_dev_max; -Params.y_dev_min];
Params.h_v = Params.h_v_phys;
Params.h_y = Params.h_y_phys;

%% 8. 参考点
Params.Kp         = [20, 40, 60];
Params.y_ref_phys = [96.5, 96.7, 96.6];

if numel(Params.Kp) ~= numel(Params.y_ref_phys)
    error('Kp and y_ref_phys length mismatch.');
end
Params.Ref_points = Params.y_ref_phys(:) - Params.y_ss(1);

%% 9. MPC 权重
Params.Q_weight = 2.0 * eye(Params.n_out);
Params.R_weight = 0.1 * eye(Params.A_lv);

%% 10. 预测/控制时域
Params.P_horizon = Params.Ble;
Params.M_horizon = Params.Ble;

%% 11. 潜增量约束
Params.delta_t_max = 0.2;
Params.ddt_max     = 0.2;

%% 12. PRBS 激励
Params.prbs_amp   = 0.01 * Params.u_ss;
Params.prbs_bits  = 10;
Params.prbs_clock = 3;
Params.noise_var  = 0.01;

%% 13. 不确定集估计
Params.safety_factor = 1.2;
Params.q_level       = 0.95;
Params.unc_method    = 'quantile';   % 'quantile' | 'scenario' | 'conformal'

%% 14. Tube 反馈
% LQR 权重选取理由（审稿人 R1-Minor 18）：
%   Q_lqr = 10*I 使跟踪误差主导，R_lqr = 1 对输入温和惩罚，
%   得到的 K_bar 使闭环谱半径<1 且约束紧化不过大。
Params.tube_alpha = 0.2;
Params.Q_lqr      = 10;
Params.R_lqr      = 1;

%% 15. 额外测试开关 (审稿人 R1-Major 6,7,8)
Params.run_init_state_test = true;    % 非相同初始状态
Params.init_state_variation = 0.0;    % 0 = 相同初始状态
Params.run_mc_test         = true;    % Monte Carlo 噪声
Params.n_mc_runs           = 5;
Params.run_model_mismatch  = false;   % 模型失配测试
Params.mismatch_gain       = 1.05;    % 1.05 = 5% 增益失配

%% 16. 绘图与随机种子
Params.plot_excitation = false;
Params.plot_model      = false;
Params.plot_results    = true;
Params.rng_seed        = 26;

%% 17. 摘要
fprintf('init_params: Wood-Berry parameters initialized.\n');
fprintf('  n_arx = %d (covers 7-sample delay)\n', Params.n_arx);
fprintf('  u_ss = [%s] | y_ss = [%s]\n', ...
    num2str(Params.u_ss'), num2str(Params.y_ss'));
fprintf('  Kp = [%s] | y_ref = [%s] wt%%\n', ...
    num2str(Params.Kp), num2str(Params.y_ref_phys));
fprintf('  Validation data N_val = %d\n', Params.N_val);
end

function [num, den] = pade2(tau)
    num = [ tau^2, -6 * tau, 12 ];
    den = [ tau^2,  6 * tau, 12 ];
end
