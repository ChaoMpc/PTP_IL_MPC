function [Phi_x, x0_contrib, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N_steps, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q, S_v, P_mat, X_mean, Y_mean, x0)
% =========================================================================
% build_lifting_matrices -- 构造预测时域的提升矩阵 (Lifting Matrices)
%
% 输入参数：
%   N_steps : 预测步数（在分阶段控制中即控制段长度 N_ctrl = 60）
%   nx      : 状态维度 (A_lv * n_state_per_lv)
%   其他    : 系统矩阵及缩放/中心化参数
%
% 输出变量及维度 (基于控制段步数 N_steps)：
%   t: [t(1), ..., t(N_steps)]          (A_lv * N_steps 维决策变量)
%   x: [x(1), ..., x(N_steps)]          (nx * N_steps 维状态预测)
%   y: [y(1), ..., y(N_steps)]          (n_out * N_steps 维输出预测)
%   v: [v(1), ..., v(N_steps)]          (n_in * N_steps 维输入预测)
% =========================================================================

% 决策变量的总维度 (A_lv * N_steps)
n_state_total = A_lv * N_steps;

%% 1. 构造状态预测矩阵 Phi_x 及初始状态贡献 x0_contrib
% 满足关系: x_vec = Phi_x * t_vec + x0_contrib
Phi_x   = zeros(nx * N_steps, n_state_total);
Gamma_x = zeros(nx * N_steps, nx);

% 预先计算 A 的幂次 A^0, A^1, ..., A^N_steps
A_pow   = cell(N_steps + 1, 1);
A_pow{1} = eye(nx);
for i = 2 : N_steps + 1
    A_pow{i} = barA * A_pow{i-1};
end

for k = 1 : N_steps
    % 状态 x(k) 占据的行范围
    r = (k-1)*nx + 1 : k*nx;
    % 累积前面所有的输入 t(j) 对 x(k) 的影响
    for j = 1 : k
        c = (j-1)*A_lv + 1 : j*A_lv;
        Phi_x(r, c) = A_pow{k-j+1} * barB;
    end
    % 初始状态 x(0) 随时间的演化
    Gamma_x(r, :) = A_pow{k+1};
end
x0_contrib = Gamma_x * x0;

%% 2. 构造输出预测矩阵 Phi_y 及常数项 y0_const
% 满足关系: y_vec = Phi_y * t_vec + y0_const
C_y      = S_y * Q * barC;
Phi_y    = zeros(n_out * N_steps, n_state_total);
y0_const = zeros(n_out * N_steps, 1);

for k = 1 : N_steps
    r  = (k-1)*n_out + 1 : k*n_out;
    sr = (k-1)*nx + 1    : k*nx;
    Phi_y(r, :)  = C_y * Phi_x(sr, :);
    y0_const(r)  = Y_mean + C_y * x0_contrib(sr);
end

%% 3. 构造输入预测矩阵 Phi_v 及常数项 v0_const
% 满足关系: v_vec = Phi_v * t_vec + v0_const
M_v      = S_v * P_mat;
Phi_v    = zeros(n_in * N_steps, n_state_total);
v0_const = zeros(n_in * N_steps, 1);

for k = 1 : N_steps
    r = (k-1)*n_in + 1 : k*n_in;
    c = (k-1)*A_lv + 1 : k*A_lv;

    % 当前时刻的输入 v(k) 仅受当前时刻的决策变量 t(k) 影响
    Phi_v(r, c) = M_v;
    v0_const(r) = X_mean;
end

end
