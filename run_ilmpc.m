function Data = run_ilmpc(Identification_data, Model, Tube, UncSets, Params)
% =========================================================================
% run_ilmpc -- 分阶段 Tube IL-MPC (基于增量 Delta_t 的迭代学习)
% =========================================================================

%% 1. Unpack Parameters
N = Params.Ble;
ctrl_start = Params.ctrl_start;
N_ctrl = N - ctrl_start;
N_batches = Params.N_batches;
n_in  = Params.n_in_opt; % 5 (优化维度)
n_in_full = Params.n_in; % 6 (物理维度)
n_out = Params.n_out;
A_lv  = Params.A_lv;
n_arx = Params.n_arx;
n_state_per_lv = 2 * n_arx - 1;

barA = Model.barA; barB = Model.barB; barC = Model.barC;
Q = Model.Q; P_mat = Model.P;
K = Tube.K; delta_v = Tube.delta_v; delta_y = Tube.delta_y;

X_mean = Identification_data.X_mean(:);
Y_mean = Identification_data.Y_mean(:);
S_v = diag(Model.X_std);
S_y = diag(Model.Y_std);

nx = size(barA, 1);
n_t = A_lv * N_ctrl;

%% 2. Build Lifting Matrices
x0_ctrl = zeros(nx, 1);
[Phi_x, x0_contrib, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N_ctrl, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q, S_v, P_mat, X_mean, Y_mean, x0_ctrl);

%% 3. Compute Initial Feasible Solution (作为第 0 批的 t_prev)
fprintf('Computing initial feasible solution...\n');
[t_prev_full, ~, ~, ~, ~] = Initial_Feasible(x0_ctrl, Identification_data, Model, Tube, Params);
if size(t_prev_full, 2) >= N_ctrl
    t_prev = reshape(t_prev_full(:, 1:N_ctrl), n_t, 1);
else
    t_prev = zeros(n_t, 1);
end

%% 4. Initialize Storage
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

%% 5. Build QP Matrices (基于决策变量 Delta_t)
% --- 5.1 构造跟踪 Mask 矩阵 ---
n_kp = length(Params.Kp);
n_track_rows = n_kp * n_out;
Phi_y_track = zeros(n_track_rows, n_t);
y_dev_target = zeros(n_track_rows, 1);
y0_const_track = zeros(n_track_rows, 1);

for j = 1:n_kp
    k_idx = Params.Kp(j);
    if k_idx < ctrl_start || k_idx > N, continue; end
    k_rel = k_idx - ctrl_start;
    r_start = (k_rel - 1) * n_out + 1;
    r_end   = k_rel * n_out;
    Phi_y_k = Phi_y(r_start:r_end, :);

    W_k = zeros(n_out, n_out);
    W_k(Params.track_idx, Params.track_idx) = sqrt(Params.Q_weight(Params.track_idx, Params.track_idx));
    target_vec = zeros(n_out, 1);
    target_vec(Params.track_idx) = Params.Ref_points(j);

    row_idx_start = (j-1)*n_out + 1;
    row_idx_end = j*n_out;
    Phi_y_track(row_idx_start:row_idx_end, :) = W_k * Phi_y_k;
    y_dev_target(row_idx_start:row_idx_end) = W_k * target_vec;

    ry = (k_rel - 1) * n_out + (1:n_out);
    y0_const_track(row_idx_start:row_idx_end) = W_k * y0_const(ry);
end

% --- 5.2 构造 Hessian 和 差分惩罚 ---
% 差分矩阵 (59*A_lv) x (60*A_lv)
D_diff = kron([-eye(N_ctrl-1), zeros(N_ctrl-1, 1)] + [zeros(N_ctrl-1, 1), eye(N_ctrl-1)], eye(A_lv));

R_blk = Params.R_weight;
if isscalar(R_blk), R_blk = R_blk * eye(A_lv); end
R_diff = kron(eye(N_ctrl-1), R_blk);

% 目标函数 H = 2 * (Phi'Q Phi + D' R D)
H_qp = 2 * (Phi_y_track' * Phi_y_track + D_diff' * R_diff * D_diff);
H_qp = (H_qp + H_qp') / 2 + 1e-8 * eye(n_t); % 保证正定

% --- 5.3 构造约束条件 ---
% (1) 物理约束 A_phys * Delta_t <= b_phys
H_v_blk = kron(eye(N_ctrl), Tube.H_v); % (10*60) x (5*60)
H_y_blk = kron(eye(N_ctrl), Tube.H_y); % (12*60) x (6*60)
A_phys = [H_v_blk * Phi_v; H_y_blk * Phi_y];

% (2) 增量约束 |Delta_t| <= delta_t_max
A_incr = [eye(n_t); -eye(n_t)];
b_incr = [Params.delta_t_max * ones(n_t, 1); Params.delta_t_max * ones(n_t, 1)];

% (3) 差分约束 |D_diff * Delta_t| <= ddt_max
A_diff = [D_diff; -D_diff];
b_diff = [Params.ddt_max * ones(size(D_diff, 1), 1); Params.ddt_max * ones(size(D_diff, 1), 1)];

% 合并所有约束
Aineq = [A_phys; A_incr; A_diff];
row_max = max(abs(Aineq), [], 2);
row_max(row_max < 1e-10) = 1;
Aineq_scaled = Aineq ./ row_max;

% 预计算物理约束中不依赖于 t_prev 的常数项
h_v_all = []; h_y_all = [];
for k = 1:N_ctrl
    k_abs = ctrl_start + k;
    h_v_all = [h_v_all; Params.u_phys_max(1:n_in) - Params.u_base(1:n_in, k_abs); ...
                       -Params.u_phys_min(1:n_in) + Params.u_base(1:n_in, k_abs)];
    h_y_all = [h_y_all; Params.y_phys_max - Params.y_base(:, k_abs + 1); ...
                       -Params.y_phys_min + Params.y_base(:, k_abs + 1)];
end
delta_v_blk = kron(ones(N_ctrl,1), delta_v);
delta_y_blk = kron(ones(N_ctrl,1), delta_y);
b_phys_const = [h_v_all - delta_v_blk; h_y_all - delta_y_blk];

%% 6. Batch Iteration
for b = 1:N_batches
    fprintf('\n=== Batch %d/%d ===\n', b, N_batches);

    % --- 6.1 求解 QP ---
    y_prev_track = Phi_y_track * t_prev + y0_const_track;

    % 目标函数梯度 f = 2 * Phi' * (y_prev - y_target)
    f_qp = 2 * Phi_y_track' * (y_prev_track - y_dev_target);

    % 平移物理约束的右端项：b_phys = b_phys_const - A_phys * t_prev
    b_phys = b_phys_const - A_phys * t_prev;

    bineq = [b_phys; b_incr; b_diff];
    bineq_scaled = bineq ./ row_max;

    tic;
    if exist('OCTAVE_VERSION', 'builtin')
        [dt_opt, ~, info] = qp(zeros(n_t, 1), H_qp, f_qp, [], [], ...
            -inf(n_t, 1), inf(n_t, 1), -inf(size(bineq_scaled)), Aineq_scaled, bineq_scaled);
        if info.info ~= 0, warning('QP failed at batch %d.', b); dt_opt = zeros(n_t, 1); end
    else
        opts = optimoptions('quadprog', 'Display', 'off', 'MaxIterations', 1000);
        [dt_opt, ~, flag] = quadprog(H_qp, f_qp, Aineq_scaled, bineq_scaled, [], [], [], [], [], opts);
        if flag <= 0, warning('QP failed at batch %d.', b); dt_opt = zeros(n_t, 1); end
    end
    Data.QP_times(b) = toc;

    % --- 6.2 还原绝对名义轨迹 ---
    t_opt = t_prev + dt_opt;
    T_new = reshape(t_opt, A_lv, N_ctrl);
    Data.T_hist(:, :, b) = T_new;

    x_nom_traj = reshape(Phi_x * t_opt + x0_contrib, nx, N_ctrl);
    y_nom_traj = reshape(Phi_y * t_opt + y0_const, n_out, N_ctrl);
    v_nom_traj = reshape(Phi_v * t_opt + v0_const, n_in, N_ctrl);

    Data.Y_nominal_hist(:, 1:ctrl_start+1, b+1) = Params.y_base(:, 1:ctrl_start+1);
    Data.V_nominal_hist(:, 1:ctrl_start, b+1) = Params.u_base(1:n_in, 1:ctrl_start);
    for k = 1:N_ctrl
        k_abs = ctrl_start + k;
        Data.Y_nominal_hist(:, k_abs+1, b+1) = Params.y_base(:, k_abs+1) + y_nom_traj(:, k);
        Data.V_nominal_hist(:, k_abs, b+1) = Params.u_base(1:n_in, k_abs) + v_nom_traj(:, k);
    end

    % --- 6.3 物理仿真 (前馈 + 状态估计 + Tube 反馈) ---
    x_plant = Params.x0_nom;
    Y_actual = zeros(n_out, N+1);
    V_actual = zeros(n_in, N);

    T0 = Params.x0_nom(1); C0 = Params.x0_nom(3);
    Y_actual(:, 1) = [T0; T0+0.08; T0-0.06; C0; max(0, C0+2e-4+1.5e-4*(T0-300)); max(0, C0-1.5e-4-1e-4*(T0-300))];

    % (1) 开环阶段
    for k = 1:ctrl_start
        v_k_full = Params.u_base(:, k);
        V_actual(:, k) = v_k_full(1:n_in);
        [y_next, x_plant] = cryst_plant(x_plant, v_k_full, Params, true);
        Y_actual(:, k+1) = y_next;
    end

    % (2) 初始化历史窗口 (用于估计 x_actual)
    u_score_hist = zeros(A_lv, n_arx);
    t_score_hist = zeros(A_lv, n_arx);
    for k = 1:ctrl_start
        v_dev_k = Params.u_base(1:n_in, k) - Params.u_base(1:n_in, k);
        v_std_k = (v_dev_k - X_mean) ./ Model.X_std;
        t_score_k = Model.R_x' * v_std_k;

        y_dev_k = Y_actual(:, k+1) - Params.y_base(:, k+1);
        y_std_k = (y_dev_k - Y_mean) ./ Model.Y_std;
        u_score_k = Model.R_u' * y_std_k;

        u_score_hist = [u_score_k, u_score_hist(:, 1:end-1)];
        t_score_hist = [t_score_k, t_score_hist(:, 1:end-1)];
    end

    % (3) 闭环阶段
    for k = 1:N_ctrl
        k_abs = ctrl_start + k;

        % A. 用历史窗口拼接估计的真实状态 x_actual
        x_actual = zeros(nx, 1);
        for a = 1:A_lv
            idx = (a-1)*n_state_per_lv + (1:n_state_per_lv);
            x_actual(idx(1:n_arx)) = u_score_hist(a, :)';
            x_actual(idx(n_arx+1:end)) = t_score_hist(a, 1:end-1)';
        end

        % B. Tube 反馈补偿
        x_nom_k = x_nom_traj(:, k);
        e_k = x_actual - x_nom_k;
        t_nom_k = T_new(:, k);
        t_actual_k = t_nom_k + K * e_k;

        % C. 转化为真实绝对物理输入
        x0_std = P_mat * t_actual_k;
        v_k_dev = S_v * x0_std + X_mean;
        v_k_abs = Params.u_base(1:n_in, k_abs) + v_k_dev;
        v_k_abs = max(min(v_k_abs, Params.u_phys_max(1:n_in)), Params.u_phys_min(1:n_in));

        v_k_full = zeros(n_in_full, 1);
        v_k_full(1:n_in) = v_k_abs;
        v_k_full(6) = 0;

        V_actual(:, k_abs) = v_k_abs;

        % =========================================================
        % ★ 新增：注入未建模扰动（与 Xue 方法完全一致）
        % 在第 15、16 批次的 80~85 分钟，强制让浓度突变，模拟外部干扰
        % =========================================================
        if k_abs >= 80 && k_abs <= 85 && b >= 15 && b <= 16
            x_plant(3) = x_plant(3) + 0.0015;
        end


        % D. 物理引擎演化
        [y_next, x_plant] = cryst_plant(x_plant, v_k_full, Params, true);
        Y_actual(:, k_abs+1) = y_next;

        % E. 更新历史窗口
        % 这里 t 用的是算出来的 t_actual，y 用测量到的 y_next
        y_dev_new = y_next - Params.y_base(:, k_abs+1);
        y_std_new = (y_dev_new - Y_mean) ./ Model.Y_std;
        u_score_new = Model.R_u' * y_std_new;

        u_score_hist = [u_score_new, u_score_hist(:, 1:end-1)];
        t_score_hist = [t_actual_k, t_score_hist(:, 1:end-1)];
    end

    % --- 6.4 存储与误差统计 ---
    Data.Y_hist(:, :, b) = Y_actual;
    Data.V_hist(:, :, b) = V_actual;

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

    t_prev = t_opt; % 更新给下一批
end
fprintf('\nrun_ilmpc: Completed %d batches.\n', N_batches);
end
