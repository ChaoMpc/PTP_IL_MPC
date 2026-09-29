function [Data] = run_ilmpc(Identification_data, Model, Tube, UncSets, Params)
    % ========================================================================
    % run_ilmpc -- Point-to-point ILC-MPC main loop.
    %
    % ------------------------- Time convention ----------------------------
    %   Batch has N steps, index k = 1..N.
    %     Output       y(k),   k = 1..N   stored in Y_batch column 1..N
    %     Latent input t(k),   k = 1..N-1 stored in T_batch column 1..N-1
    %     Phys.  input v(k),   k = 1..N-1 stored in V_batch column 1..N-1
    %     State        x(k),   k = 1..N   with x(1) = x_init
    %     Recursion    x(k+1) = barA*x(k) + barB*t(k),  k = 1..N-1
    %     Output       u(k) = barC*x(k), y(k) = Y_mean + S_y*Q*u(k)
    %   Thus t(k) affects y(k+1); y(1) is determined by the initial state.
    %
    % ------------------------- ILC incremental variables ------------------
    %   Delta_x(k)       = x_nominal^b(k) - x_nominal^{b-1}(k)
    %   Delta_t(k)       = t_nominal^b(k) - t_nominal^{b-1}(k)
    %   delta_Delta_t(k) = Delta_t(k) - Delta_t(k-1)
    %
    %   Augmented state:  Delta_x_aug(k) = [Delta_x(k); Delta_t(k-1)]
    %   Augmented dynamics:
    %       Delta_x_aug(k+1) = A_aug*Delta_x_aug(k) + B_aug*delta_Delta_t(k)
    %       Delta_u(k)       = C_aug1 * Delta_x_aug(k)
    %       Delta_t(k-1)     = C_aug2 * Delta_x_aug(k)
    %
    % ------------------------- Tube feedback ------------------------------
    %   e(k)        = x_actual(k) - x_nominal(k)       (same time index)
    %   t_actual(k) = t_nominal(k) + K_bar * e(k)
    %   v_actual(k) = X_mean + S_v * pinv(R_x') * t_actual(k)
    %
    % ------------------------- QP decision variables ----------------------
    %   d = [delta_Delta_t(k); ...; delta_Delta_t(k+M_eff-1)]
    %   Y_pred = Y_prev + F_init + G_m * d
    %   Input  constraints: H_v_blk * (F_v_init + V_prev + G_v_m * d) <= h_v_tight
    %   Output constraints: H_y_blk * (Y_prev + F_init + G_m * d)     <= h_y_tight
    % ========================================================================

    %% ==================== 1. Basic parameters ====================
    N = Params.Ble;
    A_lv = Params.A_lv;
    n_in = Params.n_in;
    n_out = Params.n_out;
    n_arx = Params.n_arx;
    P_hor = Params.P_horizon;
    M_hor = Params.M_horizon;
    N_batches = Params.N_batches;

    if P_hor > N - 1, P_hor = N - 1; end
    if M_hor > P_hor, M_hor = P_hor; end

    %% ==================== 2. Model matrices ====================
    barA = Model.barA;
    barB = Model.barB;
    barC = Model.barC;
    A_aug = Model.A_aug;
    B_aug = Model.B_aug;
    C_aug1 = Model.C_aug1;
    C_aug2 = Model.C_aug2;

    %% ==================== 3. PLS mapping and normalization ====================
    Q_pls = Model.Q;
    Q_pls_inv = pinv(Q_pls);
    R_x = Model.R_x;
    R_x_inv = pinv(R_x'); % (A_lv x n_in)

    X_mean = Identification_data.X_mean(:);
    X_std = Identification_data.X_std(:);
    Y_mean = Identification_data.Y_mean(:);
    Y_std = Identification_data.Y_std(:);
    S_v = diag(X_std);
    S_y = diag(Y_std);

    %% ==================== 4. Tube feedback and tightened constraints ====================
    K_bar = Tube.K;
    h_v_tight = Tube.h_v_tight;
    h_y_tight = Tube.h_y_tight;
    H_v = Params.H_v;
    H_y = Params.H_y;

    if max(abs(eig(barA + barB * K_bar))) >= 1
        error('run_ilmpc: Tube gain K_bar does not make the closed loop Schur stable.');
    end

    %% ==================== 5. QP weights ====================
    Q_w = Params.Q_weight;
    R_w = Params.R_weight;

    if ~isequal(size(Q_w), [n_out, n_out])
        error('run_ilmpc: Params.Q_weight must be n_out-by-n_out.');
    end

    if ~isequal(size(R_w), [A_lv, A_lv])
        error('run_ilmpc: Params.R_weight must be A_lv-by-A_lv.');
    end

    %% ==================== 6. Reference mask ====================
    mask = zeros(n_out, N);
    ref_traj = zeros(n_out, N);

    for i = 1:numel(Params.Kp)
        k_idx = Params.Kp(i);

        if k_idx < 1 || k_idx > N || floor(k_idx) ~= k_idx
            error('run_ilmpc: reference index %d out of [1, N].', k_idx);
        end

        mask(1, k_idx) = 1;
        ref_traj(1, k_idx) = Params.Ref_points(i);
    end

    %% ==================== 7. Actual plant ====================
    System_d = ss(Params.sys_d);
    System_d_d = ss(Params.sys_d_d);
    Noise_var = Params.noise_var;

    %% ==================== 8. Storage initialization ====================
    Y_hist = zeros(n_out, N, N_batches);
    V_hist = zeros(n_in, N - 1, N_batches);
    Error_norm_actual = zeros(N_batches, 1);
    Error_norm_nominal = zeros(N_batches, 1);
    Y_nominal_hist = zeros(n_out, N, N_batches + 1);
    V_nominal_hist = zeros(n_in, N - 1, N_batches + 1);
    T_nominal_hist = zeros(A_lv, N - 1, N_batches + 1);

    %% ==================== 9. Initial feasible trajectory ====================
    x_nominal_init = zeros(size(barA, 1), 1);

    [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params);


% 计算 batch 1 初始可行解对应的输出
y_feasible_1 = Y_mean + S_y*Q_pls*(Model.barC * x_feasible(:, 2:end));
fprintf('--- Initial_Feasible output vs reference points ---\n');
for i = 1:numel(Params.Kp)
    k_idx = Params.Kp(i);
    err = y_feasible_1(1, k_idx-1) - Params.Ref_points(i);
    fprintf('  k=%d: y_feas = %.4f, ref = %.4f, err = %.4f\n', ...
        k_idx, y_feasible_1(1, k_idx-1), Params.Ref_points(i), err);
end

    y_nominal_prev_batch = zeros(n_out, N);
    y_nominal_prev_batch(:, 1) = Y_mean + S_y * Q_pls * (barC * x_nominal_init);
    y_nominal_prev_batch(:, 2:N) = y_feasible;

    t_nominal_prev_batch = t_feasible;
    v_nominal_prev_batch = v_feasible;

    u_nominal_prev_batch = zeros(A_lv, N);
    u_nominal_prev_batch(:, 1) = barC * x_nominal_init;
    u_nominal_prev_batch(:, 2:N) = u_feasible;

    x_nominal_prev_batch = x_feasible;

    Y_nominal_hist(:, :, 1) = y_nominal_prev_batch;
    V_nominal_hist(:, :, 1) = v_nominal_prev_batch;
    T_nominal_hist(:, :, 1) = t_nominal_prev_batch;
    qp_fail_count = 0;

    %% ==================== 10. Batch loop ====================
    for batch = 1:N_batches

        fprintf('\n=========== Batch %d / %d ===========\n', batch, N_batches);

        Y_batch = zeros(n_out, N);
        Y_nominal_batch = zeros(n_out, N);
        V_batch = zeros(n_in, N - 1);
        V_nominal_batch = zeros(n_in, N - 1);
        T_batch = zeros(A_lv, N - 1);
        T_nominal_batch = zeros(A_lv, N - 1);
        U_batch = zeros(A_lv, N);
        U_nominal_batch = zeros(A_lv, N);
        x_nominal_batch = zeros(size(barA, 1), N);

        % Nominal initial state x(1)
        X_nominal = x_nominal_init;
        x_nominal_batch(:, 1) = X_nominal;

        u_1 = barC * X_nominal;
        U_nominal_batch(:, 1) = u_1;
        Y_nominal_batch(:, 1) = Y_mean + S_y * Q_pls * u_1;

        % Actual initial state (reset to nominal for simplicity)
        X_actual = x_nominal_init;

        % Actual system initial condition consistent with the nominal system
        DC_gain = System_d.c * ((eye(size(System_d.a)) - System_d.a) \ System_d.b);
        v_init = DC_gain \ Y_mean;

        % Attention: System_state is not the PLS model state
        System_state = (eye(size(System_d.a)) - System_d.a) \ (System_d.b * v_init);
        System_state_dis = zeros(size(System_d_d.a, 1), 1);

        y_actual_1 = System_d.c * System_state + System_d_d.c * System_state_dis;
        Y_batch(:, 1) = y_actual_1;
        U_batch(:, 1) = Q_pls_inv * pinv(S_y) * (y_actual_1 - Y_mean);

        % Incremental initial values: Delta_x(1) = 0, Delta_t(0) = 0
        error_state = zeros(size(barA, 1), 1);
        delta_t_prev = zeros(A_lv, 1);

        %% -------- Time loop k = 1..N-1 --------
        for k = 1:N - 1

            P_eff = min(P_hor, N - k);
            M_eff = min(M_hor, P_eff);
            n_var = A_lv * M_eff;

            % Augmented state [Delta_x(k); Delta_t(k-1)]
            Initial_Aug_state = [error_state; delta_t_prev];

            %% ---- 10.1 Output sensitivity matrix G_m ----
            G_m = zeros(n_out * P_eff, n_var);

            for i = 0:P_eff - 1
                row = i * n_out + 1:(i + 1) * n_out;

                for j = 0:min(i, M_eff - 1)
                    col = j * A_lv + 1:(j + 1) * A_lv;
                    G_m(row, col) = S_y * Q_pls * C_aug1 * (A_aug^(i - j)) * B_aug;
                end

            end

            %% ---- 10.2 Output free response F_init ----
            F_init = zeros(n_out * P_eff, 1);

            for i = 0:P_eff - 1
                row = i * n_out + 1:(i + 1) * n_out;
                F_init(row) = S_y * Q_pls * C_aug1 * (A_aug^(i + 1)) * Initial_Aug_state;
            end

            %% ---- 10.3 Reference and mask vectors ----
            Y_prev_vec = zeros(n_out * P_eff, 1);
            Y_ref_vec = zeros(n_out * P_eff, 1);
            M_total = zeros(n_out * P_eff, n_out * P_eff);

            for i = 1:P_eff
                idx = min(k + i, N);
                row = (i - 1) * n_out + 1:i * n_out;
                Y_prev_vec(row) = y_nominal_prev_batch(:, idx);
                Y_ref_vec(row) = ref_traj(:, idx);
                M_total(row, row) = diag(mask(:, idx));
            end

            %% ---- 10.4 QP cost ----
            dev = Y_prev_vec + F_init - Y_ref_vec;
            Q_blk = kron(eye(P_eff), Q_w);
            R_blk = kron(eye(M_eff), R_w);

            H_qp = G_m' * M_total' * Q_blk * M_total * G_m + R_blk;
            f_qp = G_m' * M_total' * Q_blk * M_total * dev;
            H_qp = 0.5 * (H_qp + H_qp');

            %% ---- 10.5 Input sensitivity matrix G_v_m ----
            G_v_m = zeros(n_in * M_eff, n_var);

            for i = 0:M_eff - 1
                row = i * n_in + 1:(i + 1) * n_in;

                for j = 0:min(i, M_eff - 1)
                    col = j * A_lv + 1:(j + 1) * A_lv;
                    G_v_m(row, col) = S_v * R_x_inv * C_aug2 * (A_aug^(i - j)) * B_aug;
                end

            end

            %% ---- 10.6 Input free response F_v_init ----
            F_v_init = zeros(n_in * M_eff, 1);

            for i = 0:M_eff - 1
                row = i * n_in + 1:(i + 1) * n_in;
                F_v_init(row) = S_v * R_x_inv * C_aug2 * (A_aug^(i + 1)) * Initial_Aug_state;
            end

            %% ---- 10.7 Previous-batch v vector ----
            V_prev_vec = zeros(n_in * M_eff, 1);

            for i = 0:M_eff - 1
                idx = min(k + i, N - 1);
                row = i * n_in + 1:(i + 1) * n_in;
                V_prev_vec(row) = v_nominal_prev_batch(:, idx);
            end

            %% ---- 10.8 Inequality constraints ----
            H_v_blk = kron(eye(M_eff), H_v);
            h_v_blk = repmat(h_v_tight, M_eff, 1);
            A_in = H_v_blk * G_v_m;
            b_in = h_v_blk - H_v_blk * (F_v_init + V_prev_vec);

            Y_nom_base = Y_prev_vec + F_init;
            H_y_blk = kron(eye(P_eff), H_y);
            h_y_blk = repmat(h_y_tight, P_eff, 1);
            A_out = H_y_blk * G_m;
            b_out = h_y_blk - H_y_blk * Y_nom_base;

            A_ineq = [A_in; A_out];
            b_ineq = [b_in; b_out];

            if any(~isfinite(A_ineq(:))) || any(~isfinite(b_ineq(:)))
                warning('run_ilmpc: batch %d, k %d has non-finite constraints; using unconstrained step.', ...
                    batch, k);
                A_ineq = zeros(size(A_ineq));
                b_ineq = inf(size(b_ineq));
            end

            %% ---- latent batch input incremental constraints |Δt(k+j)| ≤ delta_t_max ----
            if isfield(Params, 'delta_t_max') &&~isempty(Params.delta_t_max)
                delta_t_max = Params.delta_t_max;

                % 下三角累加矩阵：L_blk * d 的第 j 块 = sum_{i=0}^{j} δΔt(k+i)
                L_tril = tril(ones(M_eff, M_eff));
                L_blk = kron(L_tril, eye(A_lv));

                % 上界：delta_t_prev + L_blk * d ≤ delta_t_max
                b_upper = repmat(delta_t_max * ones(A_lv, 1) - delta_t_prev, M_eff, 1);
                % 下界：-delta_t_prev - L_blk * d ≤ delta_t_max
                b_lower = repmat(delta_t_max * ones(A_lv, 1) + delta_t_prev, M_eff, 1);

                A_dt_ineq = [L_blk; -L_blk];
                b_dt_ineq = [b_upper; b_lower];

                A_ineq = [A_ineq; A_dt_ineq];
                b_ineq = [b_ineq; b_dt_ineq];
            end

            %% ---- latent time incremental input constraints |δΔt(k+j)| ≤ ddt_max ----
            if isfield(Params, 'ddt_max') &&~isempty(Params.ddt_max)
                ddt_max = Params.ddt_max;
                A_ddt = [eye(n_var); -eye(n_var)];
                b_ddt = ddt_max * ones(2 * n_var, 1);
                A_ineq = [A_ineq; A_ddt];
                b_ineq = [b_ineq; b_ddt];
            end

            %fprintf('Batch %d, k %d: size(A_ineq) = [%d, %d], n_var = %d\n', ...
    %batch, k, size(A_ineq,1), size(A_ineq,2), n_var);

            %% ---- 10.9 Solve QP (Octave) ----
            lb = -inf(n_var, 1);
            ub = inf(n_var, 1);
            x0 = zeros(n_var, 1);
            A_lb = -inf(size(A_ineq, 1), 1);

            [d_opt, ~, info, ~] = qp(...
                x0, H_qp, f_qp, ...
                [], [], ...
                lb, ub, ...
                A_lb, A_ineq, b_ineq);

            if isstruct(info) && isfield(info, 'info')
                info_code = info.info;
            else
                info_code = info;
            end

            %% ---- 10.10 Extract increments ----
            delta_delta_t = d_opt(1:A_lv);
            delta_t_k = delta_t_prev + delta_delta_t;

            %% ---- 10.11 Nominal latent input at time k ----
            t_nominal_k = t_nominal_prev_batch(:, k) + delta_t_k;
            T_nominal_batch(:, k) = t_nominal_k;

            %% ---- 10.12 Tube feedback: e_k must use x_nom(k), NOT x_nom(k+1) ----
            % X_actual currently holds x_actual(k); X_nominal currently holds x_nom(k).
            e_k = X_actual - X_nominal; % <-- correct time index
            t_actual_k = t_nominal_k + K_bar * e_k;
            T_batch(:, k) = t_actual_k;

            v_actual_k = X_mean + S_v * R_x_inv * t_actual_k;
            V_batch(:, k) = v_actual_k;

            %% ---- 10.13 Nominal state / output update (AFTER the feedback) ----
            v_nominal_k = X_mean + S_v * R_x_inv * t_nominal_k;
            V_nominal_batch(:, k) = v_nominal_k;

            X_nominal = barA * X_nominal + barB * t_nominal_k; % x_nom(k+1)
            x_nominal_batch(:, k + 1) = X_nominal;

            u_nom_kp1 = barC * X_nominal;
            U_nominal_batch(:, k + 1) = u_nom_kp1;
            Y_nominal_batch(:, k + 1) = Y_mean + S_y * Q_pls * u_nom_kp1;

            %% ---- 10.14 Actual plant simulation ----
            d_dist = Params.noise_amptitude*randn(1,1); %扰动输入
            System_state = System_d.a * System_state + System_d.b * v_actual_k;
            y1 = System_d.c * System_state;
            System_state_dis = System_d_d.a * System_state_dis + System_d_d.b * d_dist;
            y2 = System_d_d.c * System_state_dis;
            y_actual_kp1 = y1 + y2;
            Y_batch(:, k + 1) = y_actual_kp1;

            %% ---- 10.15 Recover latent output from actual output ----
            u_actual_kp1 = Q_pls_inv * pinv(S_y) * (y_actual_kp1 - Y_mean);
            U_batch(:, k + 1) = u_actual_kp1;

            %% ---- 10.16 Update actual augmented state ----
            X_tt = zeros(size(X_actual, 1), 1);

            for a = 1:A_lv
                row = (a - 1) * (2 * n_arx - 1) + 1:a * (2 * n_arx - 1);
                X_temp1 = X_actual(row);
                u_seq = X_temp1(1:n_arx - 1);
                t_seq = X_temp1(n_arx + 1:2 * n_arx - 2);
                X_tt(row) = [u_actual_kp1(a); u_seq; t_actual_k(a); t_seq];
            end

            X_actual = X_tt;

            %% ---- 10.17 Update incremental state ----
            error_state = X_nominal - x_nominal_prev_batch(:, k + 1);
            delta_t_prev = delta_t_k;
        end

        %% -------- 10.18 Batch end: error evaluation and storage --------
        Y_hist(:, :, batch) = Y_batch;
        V_hist(:, :, batch) = V_batch;

        err_actual = 0;
        err_nominal = 0;

        for i = 1:numel(Params.Kp)
            k_idx = Params.Kp(i);
            err_actual = err_actual + (Y_batch(1, k_idx) - Params.Ref_points(i))^2;
            err_nominal = err_nominal + (Y_nominal_batch(1, k_idx) - Params.Ref_points(i))^2;
        end

        Error_norm_actual(batch) = sqrt(err_actual);
        Error_norm_nominal(batch) = sqrt(err_nominal);

        % Update previous-batch nominal trajectories
        t_nominal_prev_batch = T_nominal_batch;
        u_nominal_prev_batch = U_nominal_batch;
        y_nominal_prev_batch = Y_nominal_batch;
        v_nominal_prev_batch = V_nominal_batch;
        x_nominal_prev_batch = x_nominal_batch;

        Y_nominal_hist(:, :, batch + 1) = y_nominal_prev_batch;
        V_nominal_hist(:, :, batch + 1) = v_nominal_prev_batch;
        T_nominal_hist(:, :, batch + 1) = t_nominal_prev_batch;

        fprintf('Batch %d finished: nominal err = %.6f, actual err = %.6f\n', ...
            batch, Error_norm_nominal(batch), Error_norm_actual(batch));
    end

    %% ==================== 11. Package output ====================
    Data.Y_hist = Y_hist;
    Data.V_hist = V_hist;
    Data.Y_nominal_hist = Y_nominal_hist;
    Data.V_nominal_hist = V_nominal_hist;
    Data.T_nominal_hist = T_nominal_hist;
    Data.Error_norm_actual = Error_norm_actual;
    Data.Error_norm_nominal = Error_norm_nominal;
end

% ========================================================================
% Local function: Initial_Feasible
% ------------------------------------------------------------------------
% Generate the initial feasible nominal trajectory (lifting form).
%
% Time convention:
%   N = Params.Ble; k = 1..N
%     State      x(k),  k = 1..N      with x(1) = x_nominal_init
%     Latent in  t(k),  k = 1..N-1    (decision variables)
%     Latent out u(k) = barC*x(k),    k = 1..N
%     Phys. in   v(k) = X_mean + S_v*R_x_inv*t(k),   k = 1..N-1
%     Phys. out  y(k) = Y_mean + S_y*Q*u(k),         k = 1..N
%   Recursion: x(k+1) = barA*x(k) + barB*t(k), k = 1..N-1
%
% Returns:
%   t_feasible : (A_lv  x (N-1)), column j = t(j),   j = 1..N-1
%   u_feasible : (A_lv  x (N-1)), column j = u(j+1), j = 1..N-1
%   y_feasible : (n_out x (N-1)), column j = y(j+1), j = 1..N-1
%   v_feasible : (n_in  x (N-1)), column j = v(j),   j = 1..N-1
%   x_feasible : (nx    x N    ), column j = x(j),   j = 1..N
% ========================================================================
function [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params)

    %% ==================== 1. Basic parameters ====================
    N = Params.Ble;
    A_lv = Params.A_lv;
    n_in = Params.n_in;
    n_out = Params.n_out;
    nx = length(x_nominal_init);

    barA = Model.barA;
    barB = Model.barB;
    barC = Model.barC;
    R_x = Model.R_x;
    Q = Model.Q;

    X_mean = Identification_data.X_mean(:);
    Y_mean = Identification_data.Y_mean(:);
    X_std = Identification_data.X_std(:);
    Y_std = Identification_data.Y_std(:);
    S_v = diag(X_std);
    S_y = diag(Y_std);

    R_x_inv = pinv(R_x');

    %% ==================== 2. Tightened bounds ====================
    if isfield(Tube, 'h_v_tight') &&~isempty(Tube.h_v_tight)
        h_v_tight = Tube.h_v_tight;
    else
        h_v_tight = Params.h_v;
    end

    if isfield(Tube, 'h_y_tight') &&~isempty(Tube.h_y_tight)
        h_y_tight = Tube.h_y_tight;
    else
        h_y_tight = Params.h_y;
    end

    H_v = Params.H_v;
    H_y = Params.H_y;

    %% ==================== 3. Cost weights ====================
    R_weight = Params.R_weight;

    if isscalar(R_weight)
        R_weight = R_weight * eye(A_lv);
    end

    if ~isequal(size(R_weight), [A_lv, A_lv])
        error('Initial_Feasible: R_weight must be A_lv-by-A_lv.');
    end

    %% ==================== 4. Powers of barA ====================
    A_powers = cell(N, 1);
    A_powers{1} = eye(nx);

    for i = 2:N
        A_powers{i} = barA * A_powers{i - 1};
    end

    %% ==================== 5. Lifting matrices ====================
    n_t = A_lv * (N - 1);

    % Phi_x : x_vec = Phi_x * t_vec + x0_contrib
    Phi_x = zeros(nx * (N - 1), n_t);

    for k = 1:N - 1
        row_idx = (k - 1) * nx + 1:k * nx;

        for j = 1:k
            col_idx = (j - 1) * A_lv + 1:j * A_lv;
            Phi_x(row_idx, col_idx) = A_powers{k - j + 1} * barB;
        end

    end

    Gamma_x = zeros(nx * (N - 1), nx);

    for k = 1:N - 1
        row_idx = (k - 1) * nx + 1:k * nx;
        Gamma_x(row_idx, :) = A_powers{k + 1};
    end

    x0_contrib = Gamma_x * x_nominal_init;

    % Phi_y : y(k+1) = Y_mean + C_y * x(k+1)
    C_y = S_y * Q * barC;
    Phi_y = zeros(n_out * (N - 1), n_t);
    y0_const = zeros(n_out * (N - 1), 1);

    for k = 1:N - 1
        row_idx = (k - 1) * n_out + 1:k * n_out;
        state_row = (k - 1) * nx + 1:k * nx;
        Phi_y(row_idx, :) = C_y * Phi_x(state_row, :);
        y0_const(row_idx) = Y_mean + C_y * x0_contrib(state_row);
    end

    % Phi_v : v(k) = X_mean + M_v * t(k)
    M_v = S_v * R_x_inv;
    Phi_v = zeros(n_in * (N - 1), n_t);
    v0_const = zeros(n_in * (N - 1), 1);

    for k = 1:N - 1
        row_idx = (k - 1) * n_in + 1:k * n_in;
        col_idx = (k - 1) * A_lv + 1:k * A_lv;
        Phi_v(row_idx, col_idx) = M_v;
        v0_const(row_idx) = X_mean;
    end

    %% ==================== 6. Inequality constraints ====================
    n_constr_v = size(H_v, 1);
    n_constr_y = size(H_y, 1);
    n_ineq = (N - 1) * (n_constr_v + n_constr_y);

    Aineq = zeros(n_ineq, n_t);
    bineq = zeros(n_ineq, 1);

    for k = 1:N - 1
        row_base = (k - 1) * (n_constr_v + n_constr_y);

        rows_v = row_base + 1:row_base + n_constr_v;
        Aineq(rows_v, :) = H_v * Phi_v((k - 1) * n_in + 1:k * n_in, :);
        bineq(rows_v) = h_v_tight - H_v * v0_const((k - 1) * n_in + 1:k * n_in);

        rows_y = row_base + n_constr_v + 1:row_base + n_constr_v + n_constr_y;
        Aineq(rows_y, :) = H_y * Phi_y((k - 1) * n_out + 1:k * n_out, :);
        bineq(rows_y) = h_y_tight - H_y * y0_const((k - 1) * n_out + 1:k * n_out);
    end

    % initial incremental latent input constraints
    if isfield(Params, 'delta_t_max') &&~isempty(Params.delta_t_max)
        dt_max = Params.delta_t_max;
        % 每个时刻 k 的 t(k) 约束
        A_dt = [kron(eye(N - 1), eye(A_lv)); -kron(eye(N - 1), eye(A_lv))];
        b_dt = dt_max * ones(2 * A_lv * (N - 1), 1);
        Aineq = [Aineq; A_dt];
        bineq = [bineq; b_dt];
    end

    %% ==================== 7. Cost ====================
    H_obj = kron(eye(N - 1), R_weight);
    f_obj = zeros(n_t, 1);
    H_obj = 0.5 * (H_obj + H_obj');

    %% ==================== 8. Solve QP ====================
    x0 = zeros(n_t, 1);
    lb = -inf(n_t, 1);
    ub = inf(n_t, 1);
    A_lb = -inf(size(Aineq, 1), 1);

    [x_opt, ~, info, ~] = qp(...
        x0, H_obj, f_obj, ...
        [], [], ...
        lb, ub, ...
        A_lb, Aineq, bineq);

    % Octave >= 5.0 returns info as a struct with field .info.
    if isstruct(info) && isfield(info, 'info')
        info_code = info.info;
    else
        info_code = info;
    end

    if info_code ~= 0
        warning('Initial_Feasible: QP returned info = %d; may not be converged.', info_code);
    end

    if isempty(x_opt) || numel(x_opt) ~= n_t
        error('Initial_Feasible: QP did not return a valid solution (info = %d).', info_code);
    end

    %% ==================== 9. Extract results ====================
    t_vec = x_opt;

    t_feasible = reshape(t_vec, A_lv, N - 1);

    x_vec = Phi_x * t_vec + x0_contrib;
    x_feasible = [x_nominal_init, reshape(x_vec, nx, N - 1)];

    u_feasible = zeros(A_lv, N - 1);

    for j = 1:N - 1
        u_feasible(:, j) = barC * x_feasible(:, j + 1);
    end

    y_vec = Phi_y * t_vec + y0_const;
    y_feasible = reshape(y_vec, n_out, N - 1);

    v_vec = Phi_v * t_vec + v0_const;
    v_feasible = reshape(v_vec, n_in, N - 1);

    %% ==================== 10. Constraint violation check ====================
    max_viol_v = 0;
    max_viol_y = 0;

    for j = 1:N - 1
        max_viol_v = max(max_viol_v, max(H_v * v_feasible(:, j) - h_v_tight));
        max_viol_y = max(max_viol_y, max(H_y * y_feasible(:, j) - h_y_tight));
    end

    tol = 1e-6;

    if max_viol_v > tol || max_viol_y > tol
        warning('Initial_Feasible: constraint violations (v: %.2e, y: %.2e).', ...
            max_viol_v, max_viol_y);
    else
        fprintf('Initial_Feasible: initial feasible trajectory generated (lifting).\n');
        fprintf('  Max input  violation: %.4e\n', max_viol_v);
        fprintf('  Max output violation: %.4e\n', max_viol_y);
    end

end
