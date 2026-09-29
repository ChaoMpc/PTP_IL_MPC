function Model = build_dypls(Identification_data, Params)
% =========================================================================
% build_dypls -- Build a decoupled Dynamic PLS-ARX (DyPLS-ARX) model.
%
% Strategy
% --------
%   (1) Extract latent scores (T, U) from the standardized DEVIATION data
%       (X0, Y0) using the NIPALS algorithm.
%   (2) Fit an independent ARX model for each latent channel (t_a, u_a).
%   (3) Convert each ARX model into a state-space realization.
%   (4) Assemble the block-diagonal latent dynamic system.
%   (5) Build the augmented incremental model used by the ILC-MPC layer.
%
% Deviation-form convention (Wood-Berry incremental model)
% --------------------------------------------------------
%   The upstream gen_prbs_excitation has produced data around the steady
%   state u_ss, y_ss:
%       X = Delta u = u - u_ss      (physical deviation)
%       Y = Delta y = y - y_ss      (physical deviation)
%
%   Standardization is applied to these deviations:
%       X0 = (Delta u - X_mean) ./ X_std
%       Y0 = (Delta y - Y_mean) ./ Y_std
%
%   Because the PRBS has zero mean, X_mean and Y_mean are close to zero.
%   However, no assumption is made on their exact values, so the code
%   remains correct even when a small offset is present.
%
%   The latent ARX recursion therefore models the DEVIATION dynamics:
%       u_a(k) = alpha(1)*u_a(k-1) + ... + alpha(na)*u_a(k-na)
%              + beta(1) *t_a(k-1) + ... + beta(na) *t_a(k-na)
%              + eta_a(k)
%
%   All downstream functions (estimate_uncertainty, compute_tube,
%   run_ilmpc) rely on this deviation convention.
%
% NOTE ON CONSTRAINTS
% -------------------
%   This function does NOT deal with constraints or reference points.
%   Params.h_v_phys, Params.h_y_phys and Params.Ref_points have already
%   been prepared in init_params in the PHYSICAL DEVIATION space.
%
% Latent recursion (per channel a, forward-shifted)
% -------------------------------------------------
%   u_a(k+1) = alpha(1)*u_a(k) + ... + alpha(na)*u_a(k-na+1)
%            + beta(1) *t_a(k) + ... + beta(na) *t_a(k-na+1)
%            + eta_a(k+1)
%
% State-space realization (per channel a)
% ---------------------------------------
%   State:
%       x_a(k) = [ u_a(k); u_a(k-1); ...; u_a(k-na+1);
%                  t_a(k-1); t_a(k-2); ...; t_a(k-na+1) ]
%   Transition:
%       x_a(k+1) = A_a * x_a(k) + B_a * t_a(k) + E_a * eta_a(k+1)
%       u_a(k)   = C_a * x_a(k)
%   with C_a = [1, 0, ..., 0], E_a = [1, 0, ..., 0]'.
%   Per-channel state dim: n_state = 2 * n_arx - 1
%   Total latent state dim: nx = A_lv * n_state
%
% Inputs
% ------
%   Identification_data : struct with fields
%                           X0, Y0          - standardized DEVIATION data
%                           X_mean, X_std   - input normalization
%                           Y_mean, Y_std   - output normalization
%
%   Params              : struct with fields
%                           A_lv            - number of latent variables
%                           n_arx           - ARX order
%                           N               - sample count (checked)
%                           n_out           - output dimension
%                           plot_model      - (optional) bool, default false
%                           arx_lambda      - (optional) ARX ridge weight
%
% Outputs
% -------
%   Model.barA, barB, barC, barE                - block-diagonal latent model
%   Model.A_aug, B_aug, C_aug1, C_aug2          - augmented ILC model
%   Model.T, U, W, P, Q, R_x, R_y, Theta        - PLS / ARX parameters
%   Model.barA_cell, barB_cell, barC_cell, barE_cell
%                                               - per-channel blocks
%   Model.N, n_arx, A_lv, n_out                 - metadata
% =========================================================================

    %% =====================================================================
    % 1. Unpack inputs
    % =====================================================================
    X0 = Identification_data.X0;        % N x n_x, standardized (deviation)
    Y0 = Identification_data.Y0;        % N x n_out, standardized (deviation)

    A_lv  = Params.A_lv;
    n_arx = Params.n_arx;
    N     = Params.N;
    n_out = Params.n_out;

    if isfield(Params, 'plot_model') && ~isempty(Params.plot_model)
        plot_model = logical(Params.plot_model);
    else
        plot_model = false;
    end

    %% =====================================================================
    % 2. Consistency checks
    % =====================================================================
    if isempty(X0) || isempty(Y0)
        error('build_dypls: X0 or Y0 is empty.');
    end
    if size(X0, 1) ~= size(Y0, 1)
        error('build_dypls: X0 and Y0 must have the same number of samples.');
    end
    if size(X0, 1) ~= N
        warning(['build_dypls: X0/Y0 rows (%d) differ from Params.N (%d); ' ...
                 'using actual data length.'], size(X0, 1), N);
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

    %% =====================================================================
    % 3. NIPALS extraction of latent scores
    % =====================================================================
    T = zeros(N, A_lv);
    U = zeros(N, A_lv);
    W = zeros(n_x, A_lv);
    P = zeros(n_x, A_lv);

    X_res = X0;
    Y_res = Y0;

    max_iter = 3000;
    tol      = 1e-6;
    eps_val  = 1e-12;

    for a = 1:A_lv

        if norm(X_res, 'fro') < eps_val || norm(Y_res, 'fro') < eps_val
            warning('build_dypls: residual nearly zero before LV %d.', a);
        end

        % ---- 3.1 Initialize u with largest-energy column of Y_res ----
        col_norms = sqrt(sum(Y_res.^2, 1));
        [max_col_norm, init_col] = max(col_norms);

        if isempty(max_col_norm) || max_col_norm < eps_val
            u    = zeros(N, 1);
            u(1) = 1;
            warning('build_dypls: Y_res nearly zero at LV %d.', a);
        else
            u = Y_res(:, init_col);
        end

        % ---- 3.2 NIPALS iterations ----------------------------------
        err  = inf;
        iter = 0;

        while err > tol && iter < max_iter
            iter = iter + 1;

            w = X_res' * u;
            if norm(w) < eps_val
                warning('build_dypls: X weight nearly zero at LV %d.', a);
                break;
            end
            w = w / norm(w);

            t = X_res * w;
            if norm(t) < eps_val
                warning('build_dypls: X score nearly zero at LV %d.', a);
                break;
            end

            q = Y_res' * t / (t' * t);
            if norm(q) < eps_val
                warning('build_dypls: Y loading nearly zero at LV %d.', a);
                break;
            end
            q = q / norm(q);

            u_new = Y_res * q;
            if norm(u_new) < eps_val
                warning('build_dypls: updated u nearly zero at LV %d.', a);
                break;
            end

            if u_new' * u < 0
                u_new = -u_new;
                q     = -q;
            end

            err = norm(u_new - u) / max(norm(u), eps_val);
            u   = u_new;

            if any(~isfinite(u)) || any(~isfinite(t)) || ...
               any(~isfinite(w)) || any(~isfinite(q))
                error('build_dypls: NaN/Inf in NIPALS at LV %d.', a);
            end
        end

        if iter >= max_iter && err > tol
            warning('build_dypls: NIPALS did not converge at LV %d.', a);
        end

        % ---- 3.3 Recompute t, p, q with the final u -----------------
        w = X_res' * u;

        if norm(w) < eps_val
            warning('build_dypls: final X weight nearly zero at LV %d.', a);
            w = zeros(n_x, 1);
            t = zeros(N, 1);
            p = zeros(n_x, 1);
            q = zeros(size(Y0, 2), 1);
        else
            w = w / norm(w);
            t = X_res * w;

            if norm(t) < eps_val
                warning('build_dypls: final X score nearly zero at LV %d.', a);
                t = zeros(N, 1);
                p = zeros(n_x, 1);
                q = zeros(size(Y0, 2), 1);
            else
                p = X_res' * t / (t' * t);
                q = Y_res' * t / (t' * t);
            end
        end

        T(:, a) = t;
        U(:, a) = u;
        W(:, a) = w;
        P(:, a) = p;

        % ---- 3.4 Deflation ------------------------------------------
        if norm(t) > eps_val
            X_res = X_res - t * p';
            Y_res = Y_res - t * q';
        end

        X_res(abs(X_res) < 1e-14) = 0;
        Y_res(abs(Y_res) < 1e-14) = 0;
    end

    %% =====================================================================
    % 4. Output loading Q and input projection R_x
    % =====================================================================
    reg_Q = 1e-8 * eye(A_lv);
    Q     = (Y0' * U) / (U' * U + reg_Q);
    R_y   = Y0 - U * Q';

    R_x = W * pinv(P' * W);
    if any(any(~isfinite(R_x)))
        error('build_dypls: R_x contains NaN/Inf.');
    end

    %% =====================================================================
    % 5. ARX identification and per-channel state-space realization
    % =====================================================================
    barA_cell  = cell(A_lv, 1);
    barB_cell  = cell(A_lv, 1);
    barC_cell  = cell(A_lv, 1);
    barE_cell  = cell(A_lv, 1);
    Theta_cell = cell(A_lv, 1);

    n_state = 2 * n_arx - 1;

    if isfield(Params, 'arx_lambda') && ~isempty(Params.arx_lambda)
        lambda_base = Params.arx_lambda;
    else
        lambda_base = 1e-6;
    end

    for a = 1:A_lv

        t_seq = T(:, a);
        u_seq = U(:, a);

        % ---- 5.1 Build the regression matrix ------------------------
        n_samples_arx = N - n_arx;
        Phi   = zeros(n_samples_arx, 2 * n_arx);
        Y_arx = zeros(n_samples_arx, 1);

        row = 0;
        for k = (n_arx + 1):N
            row = row + 1;
            u_lag = u_seq(k - 1 : -1 : k - n_arx);
            t_lag = t_seq(k - 1 : -1 : k - n_arx);
            Phi(row, :)   = [u_lag; t_lag]';
            Y_arx(row)    = u_seq(k);
        end

        % ---- 5.2 Ridge-regularized least squares --------------------
        PhiTPhi    = Phi' * Phi;
        reg_scale  = trace(PhiTPhi) / (2 * n_arx);
        if reg_scale < eps_val
            reg_scale = 1;
        end

        theta = (PhiTPhi + lambda_base * reg_scale * eye(2 * n_arx)) \ (Phi' * Y_arx);

        if any(~isfinite(theta))
            error('build_dypls: ARX parameter estimation failed at LV %d.', a);
        end

        Theta_cell{a} = theta;

        alpha = theta(1:n_arx);
        beta  = theta(n_arx + 1 : 2 * n_arx);

        % ---- 5.3 Per-channel state-space matrices -------------------
        A_a = zeros(n_state, n_state);
        B_a = zeros(n_state, 1);
        C_a = zeros(1, n_state);
        E_a = zeros(n_state, 1);

        A_a(1, 1:n_arx) = alpha';
        A_a(1, n_arx + 1 : 2 * n_arx - 1) = beta(2:end)';

        if n_arx > 1
            A_a(2:n_arx, 1:n_arx - 1) = eye(n_arx - 1);
        end

        B_a(1) = beta(1);
        if n_arx > 1
            B_a(n_arx + 1) = 1;
        end

        if n_arx > 2
            A_a(n_arx + 2 : 2 * n_arx - 1, ...
                n_arx + 1 : 2 * n_arx - 2) = eye(n_arx - 2);
        end

        C_a(1) = 1;
        E_a(1) = 1;

        barA_cell{a} = A_a;
        barB_cell{a} = B_a;
        barC_cell{a} = C_a;
        barE_cell{a} = E_a;
    end

    %% =====================================================================
    % 6. Block-diagonal assembly and augmented incremental model
    % =====================================================================
    Model.barA = blkdiag(barA_cell{:});
    Model.barB = blkdiag(barB_cell{:});
    Model.barC = blkdiag(barC_cell{:});
    Model.barE = blkdiag(barE_cell{:});

    nx = size(Model.barA, 1);
    nu = A_lv;

    Model.A_aug  = [ Model.barA,           Model.barB;
                     zeros(nu, nx),        eye(nu)       ];
    Model.B_aug  = [ Model.barB;
                     eye(nu)                             ];
    Model.C_aug1 = [ Model.barC,           zeros(size(Model.barC, 1), nu) ];
    Model.C_aug2 = [ zeros(nu, nx),        eye(nu)       ];

    %% =====================================================================
    % 7. Save model parameters and metadata
    % =====================================================================
    Model.barA_cell = barA_cell;
    Model.barB_cell = barB_cell;
    Model.barC_cell = barC_cell;
    Model.barE_cell = barE_cell;

    Model.T     = T;
    Model.U     = U;
    Model.W     = W;
    Model.P     = P;
    Model.Q     = Q;
    Model.R_x   = R_x;
    Model.R_y   = R_y;
    Model.Theta = Theta_cell;

    Model.X_res_final = X_res;
    Model.Y_res_final = Y_res;

    Model.N     = N;
    Model.n_arx = n_arx;
    Model.A_lv  = A_lv;
    Model.n_out = n_out;

    %% =====================================================================
    % 8. Model fidelity check (open-loop replay on standardized data)
    % =====================================================================
    Y0_simulation = latent_model_simulation(N, A_lv, n_arx, Theta_cell, ...
        T, U, Q);

    if plot_model
        figure(99); clf;
        plot(0:N - 1, Y0_simulation, 'LineWidth', 1.2);
        hold on;
        plot(0:N - 1, Y0, '--', 'LineWidth', 1.2);
        xlabel('Sample k');
        ylabel('Standardized deviation output');
        title('DyPLS-ARX model vs. true output (open-loop)');
        legend('Model output', 'True output');
        grid on;
        drawnow;
    end

    %% =====================================================================
    % 9. Summary
    % =====================================================================
    fprintf('build_dypls: DyPLS-ARX modeling completed (deviation form).\n');
    fprintf('  Latent channels A_lv   : %d\n', A_lv);
    fprintf('  ARX order n_arx        : %d\n', n_arx);
    fprintf('  Per-channel state dim  : %d\n', n_state);
    fprintf('  Total latent state dim : %d\n', nx);
end

% =========================================================================
% Local function: latent_model_simulation
% -------------------------------------------------------------------------
% Open-loop replay of the identified ARX models over the identification
% data. Only the first n_arx latent scores are seeded from the true U;
% all subsequent values are generated by the ARX recursion itself.
%
% Reconstruction to the standardized deviation output space:
%       Y0_hat = U_sim * Q'
% =========================================================================
function Y0_simulation = latent_model_simulation(N, A_lv, n_arx, ...
        Theta_cell, T, U, Q)

    U_sim = zeros(N, A_lv);
    U_sim(1:n_arx, :) = U(1:n_arx, :);

    for a = 1:A_lv
        theta = Theta_cell{a};
        alpha = theta(1:n_arx);
        beta  = theta(n_arx + 1 : 2 * n_arx);

        for k = (n_arx + 1):N
            u_lag = U_sim(k - 1 : -1 : k - n_arx, a);
            t_lag = T(k - 1 : -1 : k - n_arx, a);

            U_sim(k, a) = alpha' * u_lag + beta' * t_lag;
        end
    end

    Y0_simulation = U_sim * Q';
end
