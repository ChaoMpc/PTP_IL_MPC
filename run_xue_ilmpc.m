function Data = run_xue_ilmpc(Identification_data, Model, Params)
% =========================================================================
% run_xue_ilmpc -- Xue's LV-PTP-ILMPC (Strictly aligned with reference paper)
% =========================================================================

N = Params.Ble;
ctrl_start = Params.ctrl_start;
N_ctrl = N - ctrl_start;
N_batches = Params.N_batches;
n_in  = Params.n_in_opt; % 5
n_in_full = Params.n_in; % 6
n_out = Params.n_out;
A_lv  = Params.A_lv;

barA = Model.barA;
barB = Model.barB;
barC = Model.barC;
Q = Model.Q;
P_mat = Model.P;

X_mean = Identification_data.X_mean(:);
Y_mean = Identification_data.Y_mean(:);
S_v = diag(Model.X_std);
S_y = diag(Model.Y_std);

nx = size(barA, 1);
n_t = A_lv * N_ctrl;

x0_ctrl = zeros(nx, 1);

[Phi_x, x0_contrib, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N_ctrl, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q, S_v, P_mat, X_mean, Y_mean, x0_ctrl);

Data = struct();
Data.Y_hist = zeros(n_out, N+1, N_batches);
Data.V_hist = zeros(n_in, N, N_batches);
Data.Y_nominal_hist = zeros(n_out, N+1, N_batches + 1);
Data.V_nominal_hist = zeros(n_in, N, N_batches + 1);
Data.T_hist = zeros(A_lv, N_ctrl, N_batches);
Data.Error_at_Kp = zeros(length(Params.Kp), N_batches);
Data.Error_norm_actual = zeros(N_batches, 1);
Data.Error_norm_nominal = zeros(N_batches, 1);
Data.QP_times = zeros(N_batches, 1);

% ★ 参考轨迹更新参数
lambda = 0.6;
y_ref_traj = zeros(n_out, N_ctrl, N_batches + 1);

% 初始参考轨迹就是基准轨迹
for k = 1:N_ctrl
    k_abs = ctrl_start + k;
    y_ref_traj(:, k, 1) = Params.y_base(:, k_abs + 1);
end

% ★ 决策变量是 Delta_t，构建惩罚 delta Delta_t 的 Hessian
D_diff = kron([-eye(N_ctrl-1), zeros(N_ctrl-1, 1)] + [zeros(N_ctrl-1, 1), eye(N_ctrl-1)], eye(A_lv));
R_blk = Params.R_weight;
if isscalar(R_blk), R_blk = R_blk * eye(A_lv); end
R_diff = kron(eye(N_ctrl-1), R_blk);

% 提升矩阵（全轨迹，无 Mask 稀疏矩阵）
W_y = sqrt(Params.Q_weight);
Phi_y_track = zeros(n_out * N_ctrl, n_t);
for k = 1:N_ctrl
    ry = (k-1) * n_out + (1:n_out);
    Phi_y_track(ry, :) = W_y * Phi_y(ry, :);
end
y0_const_track = zeros(n_out * N_ctrl, 1);
for k = 1:N_ctrl
    ry = (k-1) * n_out + (1:n_out);
    y0_const_track(ry) = W_y * y0_const(ry);
end

% H_qp = 2 * (Phi'Q Phi + D' R D)
H_qp = 2 * (Phi_y_track' * Phi_y_track + D_diff' * R_diff * D_diff);
H_qp = (H_qp + H_qp') / 2 + 1e-8 * eye(n_t);

% 构造物理约束
Aineq_phys = []; bineq_phys_const = [];
for k = 1:N_ctrl
    k_abs = ctrl_start + k;
    rv = (k-1) * n_in + (1:n_in);
    ry = (k-1) * n_out + (1:n_out);

    h_v_k = [Params.u_phys_max(1:n_in) - Params.u_base(1:n_in, k_abs);
             -Params.u_phys_min(1:n_in) + Params.u_base(1:n_in, k_abs)];
    h_y_k = [Params.y_phys_max - Params.y_base(:, k_abs + 1);
             -Params.y_phys_min + Params.y_base(:, k_abs + 1)];

    Aineq_phys = [Aineq_phys; Params.H_v * Phi_v(rv, :); Params.H_y * Phi_y(ry, :)];
    bineq_phys_const = [bineq_phys_const; h_v_k - Params.H_v * v0_const(rv);
                                          h_y_k - Params.H_y * y0_const(ry)];
end

% 构造增量与差分约束
A_incr = [eye(n_t); -eye(n_t)];
b_incr = [Params.delta_t_max * ones(n_t, 1); Params.delta_t_max * ones(n_t, 1)];

A_diff = [D_diff; -D_diff];
b_diff = [Params.ddt_max * ones(size(D_diff,1), 1); Params.ddt_max * ones(size(D_diff,1), 1)];

Aineq = [Aineq_phys; A_incr; A_diff];
row_max = max(abs(Aineq), [], 2);
row_max(row_max < 1e-10) = 1;
Aineq_scaled = Aineq ./ row_max;

% 假设第 0 批的绝对轨迹为 0
t_prev = zeros(n_t, 1);

for b = 1:N_batches
    fprintf('\n=== Batch %d/%d (Xue Method) ===\n', b, N_batches);

    % ★ 1. 用上一批的实际测量值更新当前批次的参考轨迹
    if b > 1
        for k = 1:N_ctrl
            k_abs = ctrl_start + k;
            y_ref_old = y_ref_traj(:, k, b-1);
            y_actual_prev = Data.Y_hist(:, k_abs+1, b-1);

            % ILC 学习律：向实际输出方向修正
            y_ref_new = y_ref_old + lambda * (y_ref_old - y_actual_prev);

            % ★ 强制硬对齐参考点
            for j = 1:length(Params.Kp)
                if Params.Kp(j) == k_abs
                    y_ref_new(Params.track_idx) = Params.y_base(Params.track_idx, k_abs+1) + Params.Ref_points(j);
                end
            end
            y_ref_traj(:, k, b) = y_ref_new;
        end
    end

    % ★ 2. 构造基于当前参考轨迹的 y_target 向量
    y_target_vec = zeros(n_out * N_ctrl, 1);
    for k = 1:N_ctrl
        ry = (k-1) * n_out + (1:n_out);
        k_abs = ctrl_start + k;
        y_ref_dev = y_ref_traj(:, k, b) - Params.y_base(:, k_abs+1);
        y_target_vec(ry) = W_y * y_ref_dev;
    end

    % ★ 3. 求解 QP 获取增量 Delta_t
    y_prev_track = Phi_y_track * t_prev + y0_const_track;
    f_qp = 2 * Phi_y_track' * (y_prev_track - y_target_vec);

    b_phys = bineq_phys_const - Aineq_phys * t_prev;
    bineq = [b_phys; b_incr; b_diff];
    bineq_scaled = bineq ./ row_max;

    tic;
    if exist('OCTAVE_VERSION', 'builtin')
        [dt_opt, ~, info] = qp(zeros(n_t, 1), H_qp, f_qp, [], [], ...
            -inf(n_t, 1), inf(n_t, 1), -inf(size(bineq_scaled)), ...
            Aineq_scaled, bineq_scaled);
        if info.info ~= 0, warning('QP failed.'); dt_opt = zeros(n_t, 1); end
    else
        opts = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 1000);
        [dt_opt, ~, flag] = quadprog(H_qp, f_qp, Aineq_scaled, bineq_scaled, [], [], [], [], [], opts);
        if flag <= 0, warning('QP failed.'); dt_opt = zeros(n_t, 1); end
    end
    Data.QP_times(b) = toc;

    % ★ 4. 还原绝对轨迹
    t_opt = t_prev + dt_opt;
    T_new = reshape(t_opt, A_lv, N_ctrl);
    Data.T_hist(:, :, b) = T_new;

    Y_nom = reshape(Phi_y * t_opt + y0_const, n_out, N_ctrl);
    V_nom = reshape(Phi_v * t_opt + v0_const, n_in, N_ctrl);

    V_nom_phys = zeros(n_in, N_ctrl);
    Y_nom_phys = zeros(n_out, N_ctrl);

    for k = 1:N_ctrl
        k_abs = ctrl_start + k;
        V_nom_phys(:, k) = Params.u_base(1:n_in, k_abs) + V_nom(:, k);
        Y_nom_phys(:, k) = Params.y_base(:, k_abs + 1) + Y_nom(:, k);
    end

    Data.V_nominal_hist(:, 1:ctrl_start, b+1) = Params.u_base(1:n_in, 1:ctrl_start);
    Data.V_nominal_hist(:, ctrl_start+1:N, b+1) = V_nom_phys;
    Data.Y_nominal_hist(:, 1:ctrl_start+1, b+1) = Params.y_base(:, 1:ctrl_start+1);
    Data.Y_nominal_hist(:, ctrl_start+2:N+1, b+1) = Y_nom_phys;

    % ★ 5. 物理系统仿真 (开环无反馈)
    x_plant = Params.x0_nom;
    Y_actual = zeros(n_out, N+1);
    V_actual = zeros(n_in, N);

    [y0_meas, x_plant] = cryst_plant(x_plant, [], Params, true);
    Y_actual(:, 1) = y0_meas;

    for k = 1:ctrl_start
        v_k_full = Params.u_base(:, k);
        V_actual(:, k) = v_k_full(1:n_in);
        [y_next, x_plant] = cryst_plant(x_plant, v_k_full, Params, true);
        Y_actual(:, k+1) = y_next;
    end

    for k = 1:N_ctrl
        k_abs = ctrl_start + k;

        % 纯前馈执行，没有 Tube 反馈
        dt_total = T_new(:, k);
        x0_std = Model.P * dt_total;
        v_k_dev = S_v * x0_std + X_mean;
        v_k_abs = Params.u_base(1:n_in, k_abs) + v_k_dev;
        v_k_abs = max(min(v_k_abs, Params.u_phys_max(1:n_in)), Params.u_phys_min(1:n_in));

        V_actual(:, k_abs) = v_k_abs;

        v_k_full = zeros(n_in_full, 1);
        v_k_full(1:n_in) = v_k_abs;
        v_k_full(6) = 0;

        % =========================================================
        % ★ 核心注入：在此处注入未建模扰动（与 Proposed 方法完全一致）
        % 在第 15、16 批次的 80~85 分钟，强制让浓度突变
        % =========================================================
        if k_abs >= 80 && k_abs <= 85 && b >= 15 && b <= 16
            x_plant(3) = x_plant(3) + 0.0015;
        end

        [y_next, x_plant] = cryst_plant(x_plant, v_k_full, Params, true);
        Y_actual(:, k_abs + 1) = y_next;
    end

    Data.Y_hist(:, :, b) = Y_actual;
    Data.V_hist(:, :, b) = V_actual;

    % 计算真实误差
    err_nom = zeros(length(Params.Kp), 1);
    for j = 1:length(Params.Kp)
        k_idx = Params.Kp(j);
        y_ref_abs = Params.y_base(Params.track_idx, k_idx + 1) + Params.Ref_points(j);
        y_act = Y_actual(Params.track_idx, k_idx + 1);
        Data.Error_at_Kp(j, b) = abs(y_act - y_ref_abs);

        y_nom = Data.Y_nominal_hist(Params.track_idx, k_idx + 1, b+1);
        err_nom(j) = abs(y_nom - y_ref_abs);
    end

    Data.Error_norm_actual(b) = norm(Data.Error_at_Kp(:, b));
    Data.Error_norm_nominal(b) = norm(err_nom);

    fprintf('  Actual Error: %.4e | Nominal Error: %.4e | QP Time: %.4f s\n', ...
        Data.Error_norm_actual(b), Data.Error_norm_nominal(b), Data.QP_times(b));

    t_prev = t_opt;
end
fprintf('\nrun_xue_ilmpc: Completed %d batches.\n', N_batches);
end
