function Model = build_dypls(Identification_data, Params)
    % ========================================================================
    % build_dypls -- Build a decoupled dynamic PLS-ARX model.
    %
    % Strategy:
    %   (1) Standard NIPALS PLS decomposition to extract T and U;
    %   (2) Fit independent ARX models between each (t_a, u_a) pair;
    %   (3) Convert each ARX into a state-space realization;
    %   (4) Assemble the block-diagonal latent dynamic system;
    %   (5) Build the augmented incremental model (A_aug, B_aug, C_aug1, C_aug2).
    %
    % ------------------------------ Decoupled structure -------------------
    %   X0 --PLS--> T --ARX--> U --linear reconstruction--> Y0
    %
    %   X0 ~= T*P'          (PLS input reconstruction)
    %   Y0 ~= U*Q'          (unified OLS output reconstruction)
    %   u_a(k) = ARX(t_a, u_a) + eta_a(k)
    %
    % ------------------------------ ARX model -----------------------------
    %   u_a(k) = alpha(1)*u_a(k-1) + ... + alpha(na)*u_a(k-na)
    %          + beta(1) *t_a(k-1) + ... + beta(na) *t_a(k-na)
    %          + eta_a(k)
    %
    %   Forward-shifted form:
    %   u_a(k+1) = alpha(1)*u_a(k) + ... + alpha(na)*u_a(k-na+1)
    %            + beta(1) *t_a(k) + ... + beta(na) *t_a(k-na+1)
    %            + eta_a(k+1)
    %
    % ------------------------------ State space ---------------------------
    %   State (per latent channel):
    %     x_a(k) = [u_a(k); u_a(k-1); ...; u_a(k-na+1);
    %               t_a(k-1); t_a(k-2); ...; t_a(k-na+1)]
    %   Transition:
    %     x_a(k+1) = A_a*x_a(k) + B_a*t_a(k) + E_a*eta_a(k+1)
    %     u_a(k)   = C_a*x_a(k)
    %   With C_a = [1, 0, ..., 0], E_a = [1, 0, ..., 0]^T.
    %
    %   Dimensions: n_state = 2*n_arx - 1; nx = A_lv * n_state.
    %
    % ------------------------------ Inputs --------------------------------
    %   Identification_data : contains X0, Y0 and X_mean/X_std/Y_mean/Y_std
    %   Params              : contains A_lv, n_arx, N, n_out
    %                         optional plot_model (logical, default false)
    %
    % ------------------------------ Outputs -------------------------------
    %   Model.barA, barB, barC, barE
    %   Model.A_aug, B_aug, C_aug1, C_aug2
    %   Model.T, U, W, P, Q, R_x, R_y, Theta
    %   Model.barA_cell, barB_cell, barC_cell, barE_cell
    %   Model.N, n_arx, A_lv, n_out (metadata)
    % ========================================================================

    %% ==================== 1. Read inputs ====================
    X0 = Identification_data.X0;
    Y0 = Identification_data.Y0;

    A_lv = Params.A_lv;
    n_arx = Params.n_arx;
    N = Params.N;
    n_out = Params.n_out;

    if isfield(Params, 'plot_model') &&~isempty(Params.plot_model)
        plot_model = Params.plot_model;
    else
        plot_model = false;
    end

    %% ==================== 2. Consistency checks ====================
    if isempty(X0) || isempty(Y0)
        error('build_dypls: X0 or Y0 is empty.');
    end

    if size(X0, 1) ~= size(Y0, 1)
        error('build_dypls: X0 and Y0 must have the same number of samples.');
    end

    if size(X0, 1) ~= N
        warning('build_dypls: X0/Y0 rows (%d) differ from Params.N (%d); using actual data length.', ...
            size(X0, 1), N);
        N = size(X0, 1);
    end

    if A_lv <= 0 || floor(A_lv) ~= A_lv
        error('build_dypls: Params.A_lv must be a positive integer.');
    end

    if n_arx <= 0 || floor(n_arx) ~= n_arx
        error('build_dypls: Params.n_arx must be a positive integer.');
    end

    if N <= n_arx
        error('build_dypls: N (%d) must exceed n_arx (%d).', N, n_arx);
    end

    if any(any(~isfinite(X0)))
        error('build_dypls: X0 contains NaN/Inf.');
    end

    if any(any(~isfinite(Y0)))
        error('build_dypls: Y0 contains NaN/Inf.');
    end

    n_x = size(X0, 2);

    %% ==================== 3. NIPALS extraction ====================
    X_res = X0;
    Y_res = Y0;

    T = zeros(N, A_lv);
    U = zeros(N, A_lv);
    W = zeros(n_x, A_lv);
    P = zeros(n_x, A_lv);

    max_iter = 3000;
    tol = 1e-6;
    eps_val = 1e-12;

    for a = 1:A_lv

        if norm(X_res, 'fro') < eps_val || norm(Y_res, 'fro') < eps_val
            warning('build_dypls: residual nearly zero before LV %d.', a);
        end

        % ---- Initialize u with the largest-energy column of Y_res ----
        col_norms = sqrt(sum(Y_res.^2, 1));
        [max_col_norm, init_col] = max(col_norms);

        if isempty(max_col_norm) || max_col_norm < eps_val
            u = zeros(N, 1);
            u(1) = 1;
            warning('build_dypls: Y_res nearly zero at LV %d; using unit init.', a);
        else
            u = Y_res(:, init_col);
            u = u / norm(u);
        end

        % ---- NIPALS iteration ----
        err = inf;
        iter = 0;

        while err > tol && iter < max_iter
            iter = iter + 1;

            w = X_res' * u;

            if norm(w) < eps_val
                warning('build_dypls: X weight nearly zero at LV %d; abort NIPALS.', a);
                break;
            end

            w = w / norm(w);

            t_raw = X_res * w;

            if norm(t_raw) < eps_val
                warning('build_dypls: X score nearly zero at LV %d; abort NIPALS.', a);
                break;
            end

            t = t_raw / norm(t_raw);

            q = Y_res' * t;

            if norm(q) < eps_val
                warning('build_dypls: Y loading nearly zero at LV %d; abort NIPALS.', a);
                break;
            end

            q = q / norm(q);

            u_new = Y_res * q;

            if norm(u_new) < eps_val
                warning('build_dypls: updated u nearly zero at LV %d; abort NIPALS.', a);
                break;
            end

            u_new = u_new / norm(u_new);

            % Sign alignment to avoid false non-convergence
            if u_new' * u < 0
                u_new = -u_new;
                q = -q;
            end

            err = norm(u_new - u) / max(norm(u), eps_val);
            u = u_new;

            if any(~isfinite(u)) || any(~isfinite(t)) || ...
                    any(~isfinite(w)) || any(~isfinite(q))
                error('build_dypls: NaN/Inf in NIPALS at LV %d.', a);
            end

        end

        if iter >= max_iter && err > tol
            warning('build_dypls: NIPALS did not converge at LV %d (err=%.3e).', a, err);
        end

        % ---- Recompute t, p, q with final u ----
        w = X_res' * u;

        if norm(w) < eps_val
            warning('build_dypls: final X weight nearly zero at LV %d.', a);
            w = zeros(n_x, 1);
            t = zeros(N, 1);
            p = zeros(n_x, 1);
            q = zeros(size(Y0, 2), 1);
        else
            w = w / norm(w);
            t_raw = X_res * w;

            if norm(t_raw) < eps_val
                warning('build_dypls: final X score nearly zero at LV %d.', a);
                t = zeros(N, 1);
                p = zeros(n_x, 1);
                q = zeros(size(Y0, 2), 1);
            else
                t = t_raw / norm(t_raw);
                p = X_res' * t;
                q = Y_res' * t;
            end

        end

        T(:, a) = t;
        U(:, a) = u;
        W(:, a) = w;
        P(:, a) = p;

        if norm(t) > eps_val
            X_res = X_res - t * p';
            Y_res = Y_res - t * q';
        end

        X_res(abs(X_res) < 1e-14) = 0;
        Y_res(abs(Y_res) < 1e-14) = 0;
    end

    %% ==================== 4. Q and R_x ====================
    % Unified output reconstruction: Y0 ~= U*Q'
    reg_Q = 1e-8 * eye(A_lv);
    Q = (Y0' * U) / (U' * U + reg_Q);
    R_y = Y0 - U * Q';

    % Input projection: T ~= X0 * R_x
    R_x = W * pinv(P' * W);

    if any(any(~isfinite(R_x)))
        error('build_dypls: R_x contains NaN/Inf.');
    end

    %% ==================== 5. ARX identification and state space ====================
    barA_cell = cell(A_lv, 1);
    barB_cell = cell(A_lv, 1);
    barC_cell = cell(A_lv, 1);
    barE_cell = cell(A_lv, 1);
    Theta_cell = cell(A_lv, 1);

    n_state = 2 * n_arx - 1;

    for a = 1:A_lv

        t_seq = T(:, a);
        u_seq = U(:, a);

        % ---- 5.1 Build regression matrix ----
        n_samples_arx = N - n_arx;
        Phi = zeros(n_samples_arx, 2 * n_arx);
        Y_arx = zeros(n_samples_arx, 1);

        row = 0;

        for k = (n_arx + 1):N
            row = row + 1;
            u_lag = u_seq(k - 1:-1:k - n_arx);
            t_lag = t_seq(k - 1:-1:k - n_arx);
            Phi(row, :) = [u_lag; t_lag]';
            Y_arx(row) = u_seq(k);
        end

        % ---- 5.2 Ridge-regularized least squares ----
        lambda = 1e-6;
        theta = (Phi' * Phi + lambda * eye(2 * n_arx)) \ (Phi' * Y_arx);

        if any(~isfinite(theta))
            error('build_dypls: ARX parameter estimation failed at LV %d.', a);
        end

        Theta_cell{a} = theta;

        alpha = theta(1:n_arx);
        beta = theta(n_arx + 1:2 * n_arx);

        % ---- 5.3 State-space matrices ----
        A_a = zeros(n_state, n_state);
        B_a = zeros(n_state, 1);
        C_a = zeros(1, n_state);
        E_a = zeros(n_state, 1);

        % Row 1: ARX recursion
        A_a(1, 1:n_arx) = alpha';
        A_a(1, n_arx + 1:2 * n_arx - 1) = beta(2:end)';

        % Rows 2..na: output history shift
        if n_arx > 1
            A_a(2:n_arx, 1:n_arx - 1) = eye(n_arx - 1);
        end

        % New input t_a(k) enters at position na+1
        B_a(1) = beta(1);
        B_a(n_arx + 1) = 1;

        % Rows na+2..2na-1: input history shift
        if n_arx > 2
            A_a(n_arx + 2:2 * n_arx - 1, n_arx + 1:2 * n_arx - 2) = eye(n_arx - 2);
        end

        C_a(1) = 1;
        E_a(1) = 1;

        barA_cell{a} = A_a;
        barB_cell{a} = B_a;
        barC_cell{a} = C_a;
        barE_cell{a} = E_a;
    end

    %% ==================== 6. Block-diagonal assembly ====================
    Model.barA = blkdiag(barA_cell{:});
    Model.barB = blkdiag(barB_cell{:});
    Model.barC = blkdiag(barC_cell{:});
    Model.barE = blkdiag(barE_cell{:});

    nx = size(Model.barA, 1);
    nu = A_lv;

    % Augmented incremental model:
    %   Delta_x_aug(k+1) = A_aug*Delta_x_aug(k) + B_aug*delta_Delta_t(k)
    %   Delta_u(k)     = C_aug1 * Delta_x_aug(k)
    %   Delta_t(k-1)   = C_aug2 * Delta_x_aug(k)
    Model.A_aug = [Model.barA, Model.barB;
                zeros(nu, nx), eye(nu)];
    Model.B_aug = [Model.barB; eye(nu)];
    Model.C_aug1 = [Model.barC, zeros(size(Model.barC, 1), nu)];
    Model.C_aug2 = [zeros(nu, nx), eye(nu)];

    %% ==================== 7. Save model and metadata ====================
    Model.barA_cell = barA_cell;
    Model.barB_cell = barB_cell;
    Model.barC_cell = barC_cell;
    Model.barE_cell = barE_cell;

    Model.T = T;
    Model.U = U;
    Model.W = W;
    Model.P = P;
    Model.Q = Q;
    Model.R_x = R_x;
    Model.R_y = R_y;
    Model.Theta = Theta_cell;

    Model.X_res_final = X_res;
    Model.Y_res_final = Y_res;

    Model.N = N;
    Model.n_arx = n_arx;
    Model.A_lv = A_lv;
    Model.n_out = n_out;

    %% ==================== 8. Model fidelity check ====================
    Y0_simulation = latent_model_simulation(N, A_lv, n_arx, Theta_cell, ...
        Identification_data, T, U, Q);

    if plot_model
        figure(99);
        clf;
        plot(0:N - 1, Y0_simulation, 'LineWidth', 1.2);
        hold on;
        plot(0:N - 1, Y0, '--', 'LineWidth', 1.2);
        xlabel('Time (k)');
        ylabel('Standardized output');
        title('DyPLS-ARX model vs. true output');
        legend('Model output', 'True output');
        grid on;
        drawnow;
    end

    %% ==================== 9. Summary ====================
    fprintf('build_dypls: DyPLS-ARX modeling completed.\n');
end

% ========================================================================
% Local function: latent_model_simulation
% ------------------------------------------------------------------------
% Open-loop replay of identified ARX models on the identification data,
% mapping U_hat back to standardized output space via Y0_hat = U_hat * Q'.
% The first n_arx samples use true U as initial history.
% This function is for diagnostics only; it does not participate in control.
% ========================================================================
function Y0_simulation = latent_model_simulation(N, A_lv, n_arx, Theta_cell, ...
        Identification_data, T, U, Q)
    Y0 = Identification_data.Y0;

    U_sim = zeros(N, A_lv);
    U_sim(1:n_arx, :) = U(1:n_arx, :);

    for a = 1:A_lv
        theta = Theta_cell{a};
        alpha = theta(1:n_arx);
        beta = theta(n_arx + 1:2 * n_arx);

        for k = (n_arx + 1):N
            u_lag = U(k - 1:-1:k - n_arx, a);
            t_lag = T(k - 1:-1:k - n_arx, a);
            U_sim(k, a) = alpha' * u_lag + beta' * t_lag;
        end

    end

    Y0_simulation = zeros(N, size(Y0, 2));
    Y0_simulation(1:n_arx, :) = Y0(1:n_arx, :);
    Y0_simulation(n_arx + 1:N, :) = U_sim(n_arx + 1:N, :) * Q';
end
