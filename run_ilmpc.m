function Data = run_ilmpc(Identification_data, Model, Tube, UncSets, Params)
% =========================================================================
% run_ilmpc -- Point-to-point ILC-MPC online simulation loop.
%
% SCALE CONVENTION
% -------------------------------------------------------------------------
% All quantities are PHYSICAL DEVIATIONS from steady state:
%     t(k), u(k) : latent scores       (dimensionless)
%     v(k)       : u(k) - u_ss         [lb/min]
%     y(k)       : y(k) - y_ss         [wt%]
%     Ref_points : y_ref_phys - y_ss   [wt%]  (also physical deviation)
%
% The tightened bounds Tube.h_v_tight, Tube.h_y_tight are provided in the
% SAME physical deviation space by compute_tube (see its header).
%
% This function therefore performs NO unit conversion; every arithmetic
% operation is carried out in physical deviations consistently.
% =========================================================================

    %% =====================================================================
    % 0. Load control package under Octave
    % =====================================================================
    if exist('OCTAVE_VERSION', 'builtin')
        pkg load control;
    end

    %% =====================================================================
    % 1. Unpack scalar parameters
    % =====================================================================
    N         = Params.Ble;
    A_lv      = Params.A_lv;
    n_in      = Params.n_in;
    n_out     = Params.n_out;
    n_arx     = Params.n_arx;
    P_hor     = Params.P_horizon;
    M_hor     = Params.M_horizon;
    N_batches = Params.N_batches;

    if P_hor > N - 1, P_hor = N - 1; end
    if M_hor > P_hor, M_hor = P_hor; end

    if isfield(Params, 'verbose_ilmpc') && ~isempty(Params.verbose_ilmpc)
        verbose_ilmpc = logical(Params.verbose_ilmpc);
    else
        verbose_ilmpc = false;
    end

    %% =====================================================================
    % 2. Unpack model matrices
    % =====================================================================
    barA   = Model.barA;
    barB   = Model.barB;
    barC   = Model.barC;
    A_aug  = Model.A_aug;
    B_aug  = Model.B_aug;
    C_aug1 = Model.C_aug1;
    C_aug2 = Model.C_aug2;

    nx        = size(barA, 1);
    n_state_a = 2 * n_arx - 1;

    if nx ~= A_lv * n_state_a
        error(['run_ilmpc: barA dimension (%d) is inconsistent with ' ...
               'A_lv*(2*n_arx-1) = %d.'], nx, A_lv * n_state_a);
    end

    %% =====================================================================
    % 3. PLS mapping and normalization
    % =====================================================================
    Q_pls     = Model.Q;
    Q_pls_inv = pinv(Q_pls);
    R_x       = Model.R_x;
    R_x_inv   = pinv(R_x');

    X_mean = Identification_data.X_mean(:);
    X_std  = Identification_data.X_std(:);
    Y_mean = Identification_data.Y_mean(:);
    Y_std  = Identification_data.Y_std(:);
    S_v    = diag(X_std);
    S_y    = diag(Y_std);

    %% =====================================================================
    % 4. Tube feedback and tightened constraints
    % ---------------------------------------------------------------------
    % h_v_tight, h_y_tight are PHYSICAL DEVIATION bounds produced by
    % compute_tube. They share the same scale as v and y below.
    % =====================================================================
    K_bar     = Tube.K;
    h_v_tight = Tube.h_v_tight;
    h_y_tight = Tube.h_y_tight;
    H_v       = Params.H_v;
    H_y       = Params.H_y;

    if max(abs(eig(barA + barB * K_bar))) >= 1
        error(['run_ilmpc: Tube gain K_bar does not make the closed ' ...
               'loop Schur stable.']);
    end

    %% =====================================================================
    % 5. QP weights
    % =====================================================================
    Q_w = Params.Q_weight;
    R_w = Params.R_weight;

    if ~isequal(size(Q_w), [n_out, n_out])
        error('run_ilmpc: Params.Q_weight must be n_out-by-n_out.');
    end
    if ~isequal(size(R_w), [A_lv, A_lv])
        error('run_ilmpc: Params.R_weight must be A_lv-by-A_lv.');
    end

    %% =====================================================================
    % 6. Reference mask and trajectory (PHYSICAL DEVIATION)
    % =====================================================================
    mask     = zeros(n_out, N);
    ref_traj = zeros(n_out, N);

    for i = 1:numel(Params.Kp)
        k_idx = Params.Kp(i);
        if k_idx < 1 || k_idx > N || floor(k_idx) ~= k_idx
            error('run_ilmpc: reference index %d out of [1, %d].', k_idx, N);
        end
        mask(1, k_idx)     = 1;
        ref_traj(1, k_idx) = Params.Ref_points(i);
    end

    %% =====================================================================
    % 7. Actual plant
    % =====================================================================
    System_d   = ss(Params.sys_d);
    System_d_d = ss(Params.sys_d_d);
    Noise_var  = Params.noise_var;

    %% =====================================================================
    % 8. Storage allocation
    % =====================================================================
    Y_hist             = zeros(n_out, N,     N_batches);
    V_hist             = zeros(n_in,  N - 1, N_batches);
    Error_norm_actual  = zeros(N_batches, 1);
    Error_norm_nominal = zeros(N_batches, 1);
    Y_nominal_hist     = zeros(n_out, N,     N_batches + 1);
    V_nominal_hist     = zeros(n_in,  N - 1, N_batches + 1);
    T_nominal_hist     = zeros(A_lv,  N - 1, N_batches + 1);

    %% =====================================================================
    % 9. Initial feasible trajectory
    % =====================================================================
    x_nominal_init = zeros(nx, 1);

    [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params);

    y_nominal_prev_batch          = zeros(n_out, N);
    y_nominal_prev_batch(:, 1)    = Y_mean + S_y * Q_pls * (barC * x_nominal_init);
    y_nominal_prev_batch(:, 2:N)  = y_feasible;

    t_nominal_prev_batch = t_feasible;
    v_nominal_prev_batch = v_feasible;

    u_nominal_prev_batch          = zeros(A_lv, N);
    u_nominal_prev_batch(:, 1)    = barC * x_nominal_init;
    u_nominal_prev_batch(:, 2:N)  = u_feasible;

    x_nominal_prev_batch = x_feasible;

    Y_nominal_hist(:, :, 1) = y_nominal_prev_batch;
    V_nominal_hist(:, :, 1) = v_nominal_prev_batch;
    T_nominal_hist(:, :, 1) = t_nominal_prev_batch;

    qp_fail_count = 0;

    if verbose_ilmpc
        fprintf('--- Initial_Feasible diagnostics ---\n');
        fprintf('  max|t_feasible| = %.4f\n', max(abs(t_feasible(:))));
        fprintf('  delta_t_max     = %.4f\n', Params.delta_t_max);
        y_feasible_1 = Y_mean + S_y * Q_pls * (barC * x_feasible(:, 2:end));
        for i = 1:numel(Params.Kp)
            k_idx = Params.Kp(i);
            err   = y_feasible_1(1, k_idx - 1) - Params.Ref_points(i);
            fprintf('  k=%d: y_feas = %.4f, ref = %.4f, err = %+.4f\n', ...
                k_idx, y_feasible_1(1, k_idx - 1), Params.Ref_points(i), err);
        end
    end

    %% =====================================================================
    % 10. Main batch loop
    % =====================================================================
    for batch = 1:N_batches

        fprintf('\n=========== Batch %d / %d ===========\n', batch, N_batches);

        Y_batch         = zeros(n_out, N);
        Y_nominal_batch = zeros(n_out, N);
        V_batch         = zeros(n_in,  N - 1);
        V_nominal_batch = zeros(n_in,  N - 1);
        T_batch         = zeros(A_lv,  N - 1);
        T_nominal_batch = zeros(A_lv,  N - 1);
        U_batch         = zeros(A_lv,  N);
        U_nominal_batch = zeros(A_lv,  N);
        x_nominal_batch = zeros(nx, N);

        X_nominal             = x_nominal_init;
        x_nominal_batch(:, 1) = X_nominal;

        u_1 = barC * X_nominal;
        U_nominal_batch(:, 1) = u_1;
        Y_nominal_batch(:, 1) = Y_mean + S_y * Q_pls * u_1;

        X_actual = x_nominal_init;

        DC_gain = System_d.c * ((eye(size(System_d.a)) - System_d.a) \ System_d.b);
        v_init  = DC_gain \ Y_mean;
        System_state     = (eye(size(System_d.a)) - System_d.a) \ (System_d.b * v_init);
        System_state_dis = zeros(size(System_d_d.a, 1), 1);

        y_actual_1    = System_d.c * System_state + System_d_d.c * System_state_dis;
        Y_batch(:, 1) = y_actual_1;

        U_batch(:, 1) = Q_pls_inv * pinv(S_y) * (y_actual_1 - Y_mean);

        error_state  = zeros(nx, 1);
        delta_t_prev = zeros(A_lv, 1);

        for k = 1:N - 1

            P_eff = min(P_hor, N - k);
            M_eff = min(M_hor, P_eff);
            n_var = A_lv * M_eff;

            Initial_Aug_state = [error_state; delta_t_prev];

            %% ---- 10.1 Output sensitivity G_m -------------------------
            G_m = zeros(n_out * P_eff, n_var);
            for i = 0:P_eff - 1
                row = i * n_out + 1 : (i + 1) * n_out;
                for j = 0:min(i, M_eff - 1)
                    col = j * A_lv + 1 : (j + 1) * A_lv;
                    G_m(row, col) = S_y * Q_pls * C_aug1 * (A_aug^(i - j)) * B_aug;
                end
            end

            %% ---- 10.2 Output free response F_init --------------------
            F_init = zeros(n_out * P_eff, 1);
            for i = 0:P_eff - 1
                row = i * n_out + 1 : (i + 1) * n_out;
                F_init(row) = S_y * Q_pls * C_aug1 * (A_aug^(i + 1)) * Initial_Aug_state;
            end

            %% ---- 10.3 Reference and mask vectors ---------------------
            Y_prev_vec = zeros(n_out * P_eff, 1);
            Y_ref_vec  = zeros(n_out * P_eff, 1);
            M_total    = zeros(n_out * P_eff, n_out * P_eff);

            for i = 1:P_eff
                idx = min(k + i, N);
                row = (i - 1) * n_out + 1 : i * n_out;
                Y_prev_vec(row)    = y_nominal_prev_batch(:, idx);
                Y_ref_vec(row)     = ref_traj(:, idx);
                M_total(row, row)  = diag(mask(:, idx));
            end

            %% ---- 10.4 QP cost -----------------------------------------
            dev   = Y_prev_vec + F_init - Y_ref_vec;
            Q_blk = kron(eye(P_eff), Q_w);
            R_blk = kron(eye(M_eff), R_w);

            H_qp = G_m' * M_total' * Q_blk * M_total * G_m + R_blk;
            f_qp = G_m' * M_total' * Q_blk * M_total * dev;

            H_qp = 0.5 * (H_qp + H_qp');
            reg_scale = max(trace(H_qp) / max(size(H_qp, 1), 1), 1);
            H_qp = H_qp + 1e-8 * reg_scale * eye(size(H_qp, 1));

            %% ---- 10.5 Input sensitivity G_v_m -------------------------
            G_v_m = zeros(n_in * M_eff, n_var);
            for i = 0:M_eff - 1
                row = i * n_in + 1 : (i + 1) * n_in;
                for j = 0:min(i, M_eff - 1)
                    col = j * A_lv + 1 : (j + 1) * A_lv;
                    G_v_m(row, col) = S_v * R_x_inv * C_aug2 * (A_aug^(i - j)) * B_aug;
                end
            end

            %% ---- 10.6 Input free response F_v_init --------------------
            F_v_init = zeros(n_in * M_eff, 1);
            for i = 0:M_eff - 1
                row = i * n_in + 1 : (i + 1) * n_in;
                F_v_init(row) = S_v * R_x_inv * C_aug2 * (A_aug^(i + 1)) * Initial_Aug_state;
            end

            %% ---- 10.7 Previous-batch v vector -------------------------
            V_prev_vec = zeros(n_in * M_eff, 1);
            for i = 0:M_eff - 1
                idx = min(k + i, N - 1);
                row = i * n_in + 1 : (i + 1) * n_in;
                V_prev_vec(row) = v_nominal_prev_batch(:, idx);
            end

            %% ---- 10.8 Input & output inequality constraints -----------
            H_v_blk = kron(eye(M_eff), H_v);
            h_v_blk = repmat(h_v_tight, M_eff, 1);
            A_in    = H_v_blk * G_v_m;
            b_in    = h_v_blk - H_v_blk * (F_v_init + V_prev_vec);

            Y_nom_base = Y_prev_vec + F_init;
            H_y_blk    = kron(eye(P_eff), H_y);
            h_y_blk    = repmat(h_y_tight, P_eff, 1);
            A_out      = H_y_blk * G_m;
            b_out      = h_y_blk - H_y_blk * Y_nom_base;

            A_ineq = [A_in; A_out];
            b_ineq = [b_in; b_out];

            if any(~isfinite(A_ineq(:))) || any(~isfinite(b_ineq(:)))
                warning(['run_ilmpc: batch %d, k %d has non-finite ' ...
                         'constraints.'], batch, k);
                A_ineq = zeros(size(A_ineq));
                b_ineq = inf(size(b_ineq));
            end

            %% ---- 10.9 Latent batch increment bound --------------------
            if isfield(Params, 'delta_t_max') && ~isempty(Params.delta_t_max)
                delta_t_max = Params.delta_t_max;
                L_tril = tril(ones(M_eff, M_eff));
                L_blk  = kron(L_tril, eye(A_lv));
                b_upper = repmat(delta_t_max * ones(A_lv, 1) - delta_t_prev, M_eff, 1);
                b_lower = repmat(delta_t_max * ones(A_lv, 1) + delta_t_prev, M_eff, 1);
                A_ineq  = [A_ineq;  L_blk; -L_blk];
                b_ineq  = [b_ineq;  b_upper; b_lower];
            end

            %% ---- 10.10 Latent time increment bound --------------------
            if isfield(Params, 'ddt_max') && ~isempty(Params.ddt_max)
                ddt_max = Params.ddt_max;
                A_ineq  = [A_ineq;  eye(n_var); -eye(n_var)];
                b_ineq  = [b_ineq;  ddt_max * ones(2 * n_var, 1)];
            end

            %% ---- 10.11 Solve QP ---------------------------------------
            x0   = zeros(n_var, 1);
            lb   = -inf(n_var, 1);
            ub   =  inf(n_var, 1);
            A_lb = -inf(size(A_ineq, 1), 1);

            [d_opt, ~, info, ~] = qp(x0, H_qp, f_qp, ...
                                     [], [], ...
                                     lb, ub, ...
                                     A_lb, A_ineq, b_ineq);

            if isstruct(info) && isfield(info, 'info')
                info_code = info.info;
            else
                info_code = info;
            end

            if info_code ~= 0
                warning(['run_ilmpc: QP failed (batch %d, k %d, info=%d).'], ...
                    batch, k, info_code);
                d_opt = zeros(n_var, 1);
                qp_fail_count = qp_fail_count + 1;
            end

            %% ---- 10.12 Extract increments -----------------------------
            delta_delta_t = d_opt(1:A_lv);
            delta_t_k     = delta_t_prev + delta_delta_t;

            %% ---- 10.13 Nominal latent input ---------------------------
            t_nominal_k           = t_nominal_prev_batch(:, k) + delta_t_k;
            T_nominal_batch(:, k) = t_nominal_k;

            %% ---- 10.14 Tube feedback ---------------------------------
            e_k           = X_actual - X_nominal;
            t_actual_k    = t_nominal_k + K_bar * e_k;
            T_batch(:, k) = t_actual_k;

            v_actual_k    = X_mean + S_v * R_x_inv * t_actual_k;
            V_batch(:, k) = v_actual_k;

            %% ---- 10.15 Nominal state/output advance -------------------
            v_nominal_k           = X_mean + S_v * R_x_inv * t_nominal_k;
            V_nominal_batch(:, k) = v_nominal_k;

            X_nominal                 = barA * X_nominal + barB * t_nominal_k;
            x_nominal_batch(:, k + 1) = X_nominal;

            u_nom_kp1                 = barC * X_nominal;
            U_nominal_batch(:, k + 1) = u_nom_kp1;
            Y_nominal_batch(:, k + 1) = Y_mean + S_y * Q_pls * u_nom_kp1;

            %% ---- 10.16 Actual plant simulation ------------------------
            d_dist           = sqrt(Noise_var) * randn(1, 1);
            System_state     = System_d.a * System_state + System_d.b * v_actual_k;
            y1               = System_d.c * System_state;
            System_state_dis = System_d_d.a * System_state_dis + System_d_d.b * d_dist;
            y2               = System_d_d.c * System_state_dis;
            y_actual_kp1     = y1 + y2;
            Y_batch(:, k+1)  = y_actual_kp1;

            %% ---- 10.17 Recover latent score ---------------------------
            u_actual_kp1    = Q_pls_inv * pinv(S_y) * (y_actual_kp1 - Y_mean);
            U_batch(:, k+1) = u_actual_kp1;

            %% ---- 10.18 Update actual latent state ---------------------
            X_tt = zeros(nx, 1);
            for a = 1:A_lv
                row     = (a - 1) * n_state_a + 1 : a * n_state_a;
                X_temp1 = X_actual(row);
                u_seq = X_temp1(1:n_arx - 1);
                t_seq = X_temp1(n_arx + 1 : 2 * n_arx - 2);
                X_tt(row) = [u_actual_kp1(a); ...
                             u_seq; ...
                             t_actual_k(a); ...
                             t_seq];
            end
            X_actual = X_tt;

            %% ---- 10.19 Update incremental state -----------------------
            error_state  = X_nominal - x_nominal_prev_batch(:, k + 1);
            delta_t_prev = delta_t_k;

            %% ---- 10.20 Diagnostics ------------------------------------
            if verbose_ilmpc && (k == 2 || k == 10 || k == N - 1)
                fprintf(['  k=%d: |d_opt|=%.3e, |t_nom|=%.3e, ' ...
                         '|v_act|=%.3e, |e|=%.3e\n'], ...
                    k, norm(d_opt), norm(t_nominal_k), norm(v_actual_k), norm(e_k));
            end
        end

        %% ---- Batch-end bookkeeping --------------------------------
        Y_hist(:, :, batch) = Y_batch;
        V_hist(:, :, batch) = V_batch;

        err_actual  = 0;
        err_nominal = 0;
        for i = 1:numel(Params.Kp)
            k_idx        = Params.Kp(i);
            err_actual   = err_actual  + (Y_batch(1, k_idx)         - Params.Ref_points(i))^2;
            err_nominal  = err_nominal + (Y_nominal_batch(1, k_idx) - Params.Ref_points(i))^2;
        end
        Error_norm_actual(batch)  = sqrt(err_actual);
        Error_norm_nominal(batch) = sqrt(err_nominal);

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

    fprintf('Total QP failures: %d\n', qp_fail_count);

    %% =====================================================================
    % 12. Package output
    % =====================================================================
    Data.Y_hist             = Y_hist;
    Data.V_hist             = V_hist;
    Data.Y_nominal_hist     = Y_nominal_hist;
    Data.V_nominal_hist     = V_nominal_hist;
    Data.T_nominal_hist     = T_nominal_hist;
    Data.Error_norm_actual  = Error_norm_actual;
    Data.Error_norm_nominal = Error_norm_nominal;

    if isfield(Identification_data, 'u_ss')
        Data.u_ss = Identification_data.u_ss;
    else
        Data.u_ss = Params.u_ss;
    end
    if isfield(Identification_data, 'y_ss')
        Data.y_ss = Identification_data.y_ss;
    else
        Data.y_ss = Params.y_ss;
    end
end

% =========================================================================
% Local function: Initial_Feasible
% -------------------------------------------------------------------------
% Generate the initial feasible nominal trajectory (batch 0) via a lifting
% formulation. Pure QP in the latent input sequence t_vec.
%
% Cost: minimize the control effort ||t_vec||_R^2 only.
% The reference tracking cost is intentionally NOT included here, matching
% the reference implementation. Constraints are enforced exactly.
%
% All constraints use the PHYSICAL DEVIATION bounds h_v_tight / h_y_tight
% produced by compute_tube.
% =========================================================================
function [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params)

    %% ---------------- 1. Basic parameters -------------------------------
    N     = Params.Ble;
    A_lv  = Params.A_lv;
    n_in  = Params.n_in;
    n_out = Params.n_out;
    nx    = length(x_nominal_init);

    barA = Model.barA;
    barB = Model.barB;
    barC = Model.barC;
    R_x  = Model.R_x;
    Q    = Model.Q;

    X_mean = Identification_data.X_mean(:);
    Y_mean = Identification_data.Y_mean(:);
    X_std  = Identification_data.X_std(:);
    Y_std  = Identification_data.Y_std(:);
    S_v    = diag(X_std);
    S_y    = diag(Y_std);

    R_x_inv = pinv(R_x');

    %% ---------------- 2. Tightened bounds (PHYSICAL DEVIATION) ---------
    if isfield(Tube, 'h_v_tight') && ~isempty(Tube.h_v_tight)
        h_v_tight = Tube.h_v_tight;
    else
        h_v_tight = Params.h_v_phys;
    end
    if isfield(Tube, 'h_y_tight') && ~isempty(Tube.h_y_tight)
        h_y_tight = Tube.h_y_tight;
    else
        h_y_tight = Params.h_y_phys;
    end

    H_v = Params.H_v;
    H_y = Params.H_y;

    %% ---------------- 3. Cost weights -----------------------------------
    R_weight = Params.R_weight;
    if isscalar(R_weight)
        R_weight = R_weight * eye(A_lv);
    end
    if ~isequal(size(R_weight), [A_lv, A_lv])
        error('Initial_Feasible: R_weight must be A_lv-by-A_lv.');
    end

    %% ---------------- 4. Powers of barA ---------------------------------
    A_powers    = cell(N, 1);
    A_powers{1} = eye(nx);
    for i = 2:N
        A_powers{i} = barA * A_powers{i - 1};
    end

    %% ---------------- 5. Lifting matrices -------------------------------
    n_t = A_lv * (N - 1);

    Phi_x = zeros(nx * (N - 1), n_t);
    for k = 1:N - 1
        row_idx = (k - 1) * nx + 1 : k * nx;
        for j = 1:k
            col_idx = (j - 1) * A_lv + 1 : j * A_lv;
            Phi_x(row_idx, col_idx) = A_powers{k - j + 1} * barB;
        end
    end

    Gamma_x = zeros(nx * (N - 1), nx);
    for k = 1:N - 1
        row_idx = (k - 1) * nx + 1 : k * nx;
        Gamma_x(row_idx, :) = A_powers{k + 1};
    end
    x0_contrib = Gamma_x * x_nominal_init;

    % y(k+1) = Y_mean + C_y * x(k+1)
    C_y      = S_y * Q * barC;
    Phi_y    = zeros(n_out * (N - 1), n_t);
    y0_const = zeros(n_out * (N - 1), 1);
    for k = 1:N - 1
        row_idx   = (k - 1) * n_out + 1 : k * n_out;
        state_row = (k - 1) * nx + 1 : k * nx;
        Phi_y(row_idx, :) = C_y * Phi_x(state_row, :);
        y0_const(row_idx) = Y_mean + C_y * x0_contrib(state_row);
    end

    % v(k) = X_mean + M_v * t(k)
    M_v      = S_v * R_x_inv;
    Phi_v    = zeros(n_in * (N - 1), n_t);
    v0_const = zeros(n_in * (N - 1), 1);
    for k = 1:N - 1
        row_idx = (k - 1) * n_in + 1 : k * n_in;
        col_idx = (k - 1) * A_lv + 1 : k * A_lv;
        Phi_v(row_idx, col_idx) = M_v;
        v0_const(row_idx)       = X_mean;
    end

    %% ---------------- 6. Inequality constraints -------------------------
    n_constr_v = size(H_v, 1);
    n_constr_y = size(H_y, 1);
    n_ineq     = (N - 1) * (n_constr_v + n_constr_y);

    Aineq = zeros(n_ineq, n_t);
    bineq = zeros(n_ineq, 1);

    for k = 1:N - 1
        row_base = (k - 1) * (n_constr_v + n_constr_y);

        rows_v = row_base + 1 : row_base + n_constr_v;
        Aineq(rows_v, :) = H_v * Phi_v((k - 1) * n_in + 1 : k * n_in, :);
        bineq(rows_v)    = h_v_tight - H_v * v0_const((k - 1) * n_in + 1 : k * n_in);

        rows_y = row_base + n_constr_v + 1 : row_base + n_constr_v + n_constr_y;
        Aineq(rows_y, :) = H_y * Phi_y((k - 1) * n_out + 1 : k * n_out, :);
        bineq(rows_y)    = h_y_tight - H_y * y0_const((k - 1) * n_out + 1 : k * n_out);
    end

    if isfield(Params, 'delta_t_max') && ~isempty(Params.delta_t_max)
        dt_max = Params.delta_t_max;
        A_dt   = [ kron(eye(N - 1), eye(A_lv)); -kron(eye(N - 1), eye(A_lv)) ];
        b_dt   = dt_max * ones(2 * A_lv * (N - 1), 1);
        Aineq  = [Aineq; A_dt];
        bineq  = [bineq; b_dt];
    end

    %% ---------------- 7. QP diagnostics (t = 0 feasibility) -------------
    fprintf('--- Initial_Feasible QP diagnostics ---\n');
    fprintf('  n_var=%d, n_ineq=%d\n', n_t, size(Aineq,1));

    t_zero  = zeros(n_t, 1);
    y0_full = Phi_y * t_zero + y0_const;
    v0_full = Phi_v * t_zero + v0_const;
    y0_mat  = reshape(y0_full, n_out, N-1);
    v0_mat  = reshape(v0_full, n_in,  N-1);

    viol_v = H_v * v0_mat - h_v_tight;
    viol_y = H_y * y0_mat - h_y_tight;

    fprintf('  t=0  max input  violation = %+.4e\n', max(viol_v(:)));
    fprintf('  t=0  max output violation = %+.4e\n', max(viol_y(:)));
    fprintf('  h_v_tight = [%s]\n', num2str(h_v_tight(:)','  %.4e'));
    fprintf('  h_y_tight = [%s]\n', num2str(h_y_tight(:)','  %.4e'));

    %% ---------------- 8. Cost: minimize control effort only -------------
    H_obj = kron(eye(N - 1), R_weight);
    f_obj = zeros(n_t, 1);
    H_obj = 0.5 * (H_obj + H_obj');

    %% ---------------- 9. Solve QP ---------------------------------------
    x0   = zeros(n_t, 1);
    lb   = -inf(n_t, 1);
    ub   =  inf(n_t, 1);
    A_lb = -inf(size(Aineq, 1), 1);

    [x_opt, ~, info, ~] = qp(x0, H_obj, f_obj, ...
                             [], [], ...
                             lb, ub, ...
                             A_lb, Aineq, bineq);

    if isstruct(info) && isfield(info, 'info')
        info_code = info.info;
    else
        info_code = info;
    end

    if info_code ~= 0
        warning('Initial_Feasible: QP returned info = %d.', info_code);
    end

    if isempty(x_opt) || numel(x_opt) ~= n_t
        error('Initial_Feasible: QP did not return a valid solution.');
    end

    %% ---------------- 10. Extract results -------------------------------
    t_vec = x_opt;
    t_feasible = reshape(t_vec, A_lv, N - 1);

    x_vec      = Phi_x * t_vec + x0_contrib;
    x_feasible = [x_nominal_init, reshape(x_vec, nx, N - 1)];

    u_feasible = zeros(A_lv, N - 1);
    for j = 1:N - 1
        u_feasible(:, j) = barC * x_feasible(:, j + 1);
    end

    y_vec      = Phi_y * t_vec + y0_const;
    y_feasible = reshape(y_vec, n_out, N - 1);

    v_vec      = Phi_v * t_vec + v0_const;
    v_feasible = reshape(v_vec, n_in, N - 1);

    %% ---------------- 11. Constraint violation check --------------------
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
        if isfield(Params, 'verbose_ilmpc') && Params.verbose_ilmpc
            fprintf('Initial_Feasible: initial feasible trajectory generated.\n');
            fprintf('  Max input  violation: %.4e\n', max_viol_v);
            fprintf('  Max output violation: %.4e\n', max_viol_y);
        end
    end
end
