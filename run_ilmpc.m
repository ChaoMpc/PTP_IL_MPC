function Data = run_ilmpc(Identification_data, Model, Tube, UncSets, Params)
% =========================================================================
% run_ilmpc -- 依照 Revised_IL_MPC 的 Algorithm 2：
%   每批开始求解一次全时域 QP；批内执行时仅用 tube 反馈，不再重优化。
% =========================================================================

if exist('OCTAVE_VERSION', 'builtin'), pkg load control optim; end

%% 1. 参数
N = Params.Ble; A_lv = Params.A_lv; n_in = Params.n_in; n_out = Params.n_out;
n_arx = Params.n_arx; N_batches = Params.N_batches;

barA = Model.barA; barB = Model.barB; barC = Model.barC;
A_aug = Model.A_aug; B_aug = Model.B_aug;
C_aug1 = Model.C_aug1; C_aug2 = Model.C_aug2;

nx = size(barA, 1);
n_state_a = 2*n_arx - 1;
if nx ~= A_lv*n_state_a, error('run_ilmpc: state dim mismatch.'); end

Q_pls     = Model.Q;
Q_pls_inv = pinv(Q_pls);
R_x       = Model.R_x;
R_x_inv   = pinv(R_x');
X_mean = Identification_data.X_mean(:);
X_std  = Identification_data.X_std(:);
Y_mean = Identification_data.Y_mean(:);
Y_std  = Identification_data.Y_std(:);
S_v = diag(X_std); S_y = diag(Y_std);

K_bar     = Tube.K;
h_v_tight = Tube.h_v_tight;
h_y_tight = Tube.h_y_tight;
H_v = Params.H_v; H_y = Params.H_y;

if max(abs(eig(barA + barB*K_bar))) >= 1
    error('run_ilmpc: K_bar not Schur stabilizing.');
end

Q_w = Params.Q_weight; R_w = Params.R_weight;

%% 2. 掩码与参考
mask = zeros(n_out, N); ref_traj = zeros(n_out, N);
for i = 1:numel(Params.Kp)
    k_idx = Params.Kp(i);
    mask(1, k_idx) = 1;
    ref_traj(1, k_idx) = Params.Ref_points(i);
end

%% 3. 被控对象
System_d   = ss(Params.sys_d);
System_d_d = ss(Params.sys_d_d);
Noise_var  = Params.noise_var;

if isfield(Params,'run_model_mismatch') && Params.run_model_mismatch
    System_d = ss(Params.mismatch_gain * Params.sys_d);
    fprintf('run_ilmpc: model mismatch applied (gain=%.3f).\n', Params.mismatch_gain);
end

%% 4. 全时域提升矩阵
[Phi_x, x0_contrib, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q_pls, S_v, R_x_inv, X_mean, Y_mean, zeros(nx,1));

n_t   = A_lv*(N-1);
K_cum = kron(tril(ones(N-1, N-1)), eye(A_lv));
Phi_x_K = Phi_x * K_cum;
Phi_y_K = Phi_y * K_cum;
Phi_v_K = Phi_v * K_cum;

Mask_vec = zeros(n_out*(N-1), 1);
Y_ref_vec = zeros(n_out*(N-1), 1);
for i = 1:N-1
    k = i+1; row_idx = (i-1)*n_out+1 : i*n_out;
    Y_ref_vec(row_idx) = ref_traj(:, k);
    Mask_vec(row_idx)  = mask(:, k)';
end
Mask_mat = diag(Mask_vec);
Q_blk = kron(eye(N-1), Q_w);
R_blk = kron(eye(N-1), R_w);

H_qp_base = Phi_y_K' * Mask_mat * Q_blk * Mask_mat * Phi_y_K + R_blk;
H_qp_base = 0.5*(H_qp_base + H_qp_base');
H_qp_base = H_qp_base + 1e-10*eye(n_t);
G_qp_base = Phi_y_K' * Mask_mat * Q_blk * Mask_mat;

H_v_blk = kron(eye(N-1), H_v);
H_y_blk = kron(eye(N-1), H_y);
A_v_full = H_v_blk * Phi_v_K;
A_y_full = H_y_blk * Phi_y_K;

dt_max  = Params.delta_t_max;
ddt_max = Params.ddt_max;
A_dt_K  = [K_cum; -K_cum];
b_dt_K  = dt_max * ones(2*n_t, 1);
A_ddt   = [eye(n_t); -eye(n_t)];
b_ddt   = ddt_max * ones(2*n_t, 1);

A_ineq_base = [A_v_full; A_y_full; A_dt_K; A_ddt];

%% 5. 初始可行轨迹
x_nominal_init = zeros(nx, 1);
[t_feas, ~, ~, ~, ~] = Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params);
t_vec_prev = t_feas(:);
V_prev = Phi_v * t_vec_prev + v0_const;
Y_prev = Phi_y * t_vec_prev + y0_const;

%% 6. 存储
Y_hist         = zeros(n_out, N, N_batches);
V_hist         = zeros(n_in,  N-1, N_batches);
Y_nom_hist     = zeros(n_out, N, N_batches+1);
V_nom_hist     = zeros(n_in,  N-1, N_batches+1);
Err_nom        = zeros(N_batches, 1);
Err_act        = zeros(N_batches, 1);
QP_times       = zeros(N_batches, 1);

eta_in_flags = zeros(N_batches, N-1, A_lv);
r_in_flags   = zeros(N_batches, N-1, n_out);

Y_nom_hist(:, :, 1) = [Y_mean + S_y*Q_pls*barC*x_nominal_init, ...
                       reshape(Y_prev, n_out, N-1)];
V_nom_hist(:, :, 1) = reshape(V_prev, n_in, N-1);

qp_fail_count = 0;

%% 7. 主循环
for b = 1:N_batches
    fprintf('\n===== Batch %d / %d =====\n', b, N_batches);

  % ---- 7.1 初始状态（严格从不变集采样）----
    if isfield(Params, 'init_state_variation') && Params.init_state_variation > 0
        % 直接给潜状态施加偏差
        x_init_batch = x_nominal_init;
        dir_vec = randn(A_lv, 1);
        dir_vec = dir_vec / max(norm(dir_vec), eps);
        for a = 1:A_lv
            idx_u = (a-1)*n_state_a + 1;
            x_init_batch(idx_u) = x_init_batch(idx_u) + Params.init_state_variation * dir_vec(a);
        end
    else
        x_init_batch = x_nominal_init;
    end

    % ---- 7.2 求解全时域 QP ----
    b_ineq = [repmat(h_v_tight, N-1, 1) - H_v_blk * V_prev;
              repmat(h_y_tight, N-1, 1) - H_y_blk * Y_prev;
              b_dt_K;
              b_ddt];
    f_qp   = G_qp_base * (Y_prev - Y_ref_vec);

    tic;
    [ddt_opt, ~, info, ~] = qp(zeros(n_t,1), H_qp_base, f_qp, [], [], ...
        -inf(n_t,1), inf(n_t,1), -inf(size(A_ineq_base,1),1), A_ineq_base, b_ineq);
    QP_times(b) = toc;

    if isstruct(info) && isfield(info, 'info')
        info_code = info.info;
    else
        info_code = info;
    end

    if info_code ~= 0
        warning('run_ilmpc: QP failed (batch %d, info=%d); using previous trajectory.', b, info_code);
        ddt_opt = zeros(n_t, 1);
        qp_fail_count = qp_fail_count + 1;
    end

    % ---- 7.3 更新名义轨迹 ----
    t_vec_new = t_vec_prev + K_cum * ddt_opt;
    V_new = Phi_v * t_vec_new + v0_const;
    Y_new = Phi_y * t_vec_new + y0_const;
    X_new = Phi_x * t_vec_new + x0_contrib;

    T_nom = reshape(t_vec_new, A_lv, N-1);
    V_nom = reshape(V_new, n_in, N-1);
    Y_nom_stack = reshape(Y_new, n_out, N-1);
    X_nom = [x_nominal_init, reshape(X_new, nx, N-1)];

    Y_nom = [Y_mean + S_y*Q_pls*barC*x_nominal_init, Y_nom_stack];
    U_nom = [barC*x_nominal_init, barC*X_nom(:, 2:end)];

    % ---- 7.4 批内执行（tube 反馈）----
    Y_batch = zeros(n_out, N);
    V_batch = zeros(n_in,  N-1);
    U_batch = zeros(A_lv,  N);
    T_batch = zeros(A_lv,  N-1);

    % 名义状态初始化
    X_nominal = x_nominal_init;
    U_batch(:,1) = barC * X_nominal;
    Y_nominal_batch(:,1) = Y_mean + S_y * Q_pls * U_batch(:,1);

    % ★ 实际状态与物理对象初始化（注入偏差）★
    X_actual = x_init_batch;

    % 利用带有偏差的实际状态，计算出第一个时刻的实际物理输出
    U_act_1 = barC * X_actual;
    Y_batch(:,1) = Y_mean + S_y * Q_pls * U_act_1;

    % 根据第一个时刻的实际输出，反推物理对象的内部状态
    DC_gain = System_d.c * ((eye(size(System_d.a)) - System_d.a) \ System_d.b);
    v_init_actual = DC_gain \ (Y_batch(:, 1) - Y_mean);
    Sys_state     = (eye(size(System_d.a)) - System_d.a) \ (System_d.b * v_init_actual);
    Sys_state_dis = zeros(size(System_d_d.a, 1), 1);

    for k = 1:N-1
        e_k = X_actual - X_nominal;

        t_actual = T_nom(:, k) + K_bar * e_k;
        v_actual = X_mean + S_v * R_x_inv * t_actual;
        V_batch(:, k) = v_actual;
        T_batch(:, k) = t_actual; % <--- 新增：保存到历史中

        X_nominal = barA*X_nominal + barB*T_nom(:, k);

        % 实际对象推进
        d_dist = sqrt(Noise_var) * randn(1,1);
        Sys_state     = System_d.a*Sys_state + System_d.b*v_actual;
        y1            = System_d.c*Sys_state;
        Sys_state_dis = System_d_d.a*Sys_state_dis + System_d_d.b*d_dist;
        y2            = System_d_d.c*Sys_state_dis;
        y_actual_kp1  = y1 + y2;
        Y_batch(:, k+1) = y_actual_kp1;

        u_actual_kp1 = Q_pls_inv * pinv(S_y) * (y_actual_kp1 - Y_mean);
        U_batch(:, k+1) = u_actual_kp1;

        % ---- 覆盖率统计 (修复版) ----
        for a = 1:A_lv
            theta = Model.Theta{a};
            alpha = theta(1:n_arx);
            beta  = theta(n_arx+1:2*n_arx);
            if k >= n_arx
                % 使用真实的 U_batch 和 T_batch 计算历史
                u_lag = U_batch(a, k:-1:k-n_arx+1);
                t_lag = T_batch(a, k:-1:k-n_arx+1);
                eta_a = u_actual_kp1(a) - alpha'*u_lag' - beta'*t_lag';
                eta_in_flags(b, k, a) = double(abs(eta_a) <= UncSets.eta_bound(a));
            end
        end

        Y0_meas = S_y \ (y_actual_kp1 - Y_mean);
        r_k     = Y0_meas - Q_pls * u_actual_kp1;
        for i = 1:n_out
            r_in_flags(b, k, i) = double(abs(r_k(i)) <= UncSets.r_bound(i));
        end

        % 更新实际状态
        X_tt = zeros(nx,1);
        for a = 1:A_lv
            row = (a-1)*n_state_a+1 : a*n_state_a;
            X_temp1 = X_actual(row);
            u_seq = X_temp1(1:n_arx-1);
            t_seq = X_temp1(n_arx+1:2*n_arx-2);
            X_tt(row) = [u_actual_kp1(a); u_seq; t_actual(a); t_seq];
        end
        X_actual = X_tt;
    end

    % ---- 7.5 保存与误差统计 ----
    Y_hist(:, :, b) = Y_batch;
    V_hist(:, :, b) = V_batch;

    ea = 0; en = 0;
    for i = 1:numel(Params.Kp)
        k_i = Params.Kp(i);
        ea = ea + (Y_batch(1, k_i)         - Params.Ref_points(i))^2;
        en = en + (Y_nom(1, k_i)           - Params.Ref_points(i))^2;
    end
    Err_act(b) = sqrt(ea); Err_nom(b) = sqrt(en);

    t_vec_prev = t_vec_new;
    V_prev     = V_new;
    Y_prev     = Y_new;

    Y_nom_hist(:, :, b+1) = Y_nom;
    V_nom_hist(:, :, b+1) = V_nom;

    fprintf('Batch %d: nom err=%.4e, act err=%.4e, QP time=%.4fs\n', ...
        b, Err_nom(b), Err_act(b), QP_times(b));
end

%% 8. 打包
Data.Y_hist         = Y_hist;
Data.V_hist         = V_hist;
Data.Y_nominal_hist = Y_nom_hist;
Data.V_nominal_hist = V_nom_hist;
Data.Error_norm_actual  = Err_act;
Data.Error_norm_nominal = Err_nom;
Data.QP_times           = QP_times;
Data.eta_in_flags       = eta_in_flags;
Data.r_in_flags         = r_in_flags;
Data.u_ss = Identification_data.u_ss;
Data.y_ss = Identification_data.y_ss;

fprintf('run_ilmpc: simulation completed. Total QP failures: %d\n', qp_fail_count);
end
