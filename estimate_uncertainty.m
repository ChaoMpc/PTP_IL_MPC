function UncSets = estimate_uncertainty(Model, Identification_data, Params)
% =========================================================================
% estimate_uncertainty -- Estimate the two model-uncertainty bounds.
%
% Two independent uncertainty sources are quantified from the
% identification data:
%
%   (1) Dynamic ARX innovation (latent, standardized deviation domain):
%           eta_a(k) = u_a(k) - alpha_a'*u_lag - beta_a'*t_lag
%
%   (2) Output reconstruction residual (standardized deviation domain):
%           r(k) = Y0(k) - U(k)*Q'
%
% Bound estimation formula (per channel):
%
%       bound = safety_factor * column_quantile(|residual|, q_level)
%
% The resulting bounds feed the tube-MPC tightening stage:
%       - eta_bound  -> drives the error ellipsoid P and delta_v / delta_y
%       - r_bound    -> enters W_r and contributes to delta_y
%
% NOTE ON CONSTRAINT SCALE
% ------------------------
%   The residuals eta and r here are computed in the STANDARDIZED DEVIATION
%   domain (they are internal model-mismatch measures). They are consumed
%   ONLY by compute_tube, where they are converted to PHYSICAL DEVIATION
%   tube margins via S_v = diag(X_std), S_y = diag(Y_std).
%
%   This function does NOT deal with the physical constraints directly;
%   Params.h_v_phys and Params.h_y_phys live in the physical deviation
%   space and are handled by compute_tube and run_ilmpc.
%
% Inputs
% ------
%   Model               : struct returned by build_dypls, containing
%                           U, T, Theta, Q, R_y
%                           A_lv, N, n_arx, n_out
%   Identification_data : struct returned by gen_prbs_excitation
%   Params              : struct with fields
%                           N, A_lv, n_arx
%                           safety_factor  - optional, default 1.2
%                           q_level        - optional, default 0.95
%
% Outputs
% -------
%   UncSets.eta_bound     : A_lv  x 1  bounds for each ARX innovation
%   UncSets.r_bound       : n_out x 1  bounds for each output channel
%   UncSets.eta           : N_use x A_lv  ARX residual array
%   UncSets.r             : N_use x n_out output residual array
%   UncSets.safety_factor : scalar
%   UncSets.q_level       : scalar
%   UncSets.n_eta_samples : number of valid ARX residual samples
%   UncSets.n_r_samples   : number of valid output residual samples
% =========================================================================

    %% =====================================================================
    % 1. Unpack parameters
    % =====================================================================
    N     = Params.N;
    A_lv  = Params.A_lv;
    n_arx = Params.n_arx;

    U_lv  = Model.U;
    T_lv  = Model.T;
    Theta = Model.Theta;

    %% =====================================================================
    % 2. Consistency checks
    % =====================================================================
    if isempty(U_lv) || isempty(T_lv)
        error('estimate_uncertainty: Model.U or Model.T is empty.');
    end
    if size(U_lv, 1) ~= size(T_lv, 1)
        error('estimate_uncertainty: Model.U and Model.T sample counts differ.');
    end
    if size(U_lv, 2) < A_lv || size(T_lv, 2) < A_lv
        error('estimate_uncertainty: fewer latent columns than A_lv.');
    end
    if length(Theta) < A_lv
        error('estimate_uncertainty: Model.Theta has too few parameter sets.');
    end
    if any(any(~isfinite(U_lv))) || any(any(~isfinite(T_lv)))
        error('estimate_uncertainty: Model.U or Model.T contains NaN/Inf.');
    end

    if isfield(Model, 'A_lv') && ~isempty(Model.A_lv) && Model.A_lv ~= A_lv
        error(['estimate_uncertainty: Model.A_lv (%d) differs from ' ...
               'Params.A_lv (%d).'], Model.A_lv, A_lv);
    end
    if isfield(Model, 'n_arx') && ~isempty(Model.n_arx) && Model.n_arx ~= n_arx
        error(['estimate_uncertainty: Model.n_arx (%d) differs from ' ...
               'Params.n_arx (%d).'], Model.n_arx, n_arx);
    end

    if isfield(Model, 'N') && ~isempty(Model.N)
        N_model = Model.N;
    else
        N_model = N;
    end
    N_data = size(U_lv, 1);

    if ~(N == N_model && N == N_data)
        warning(['estimate_uncertainty: sample counts differ ' ...
                 '(Params.N=%d, Model.N=%d, data rows=%d). ' ...
                 'Using the minimum.'], N, N_model, N_data);
    end
    N_use = min([N, N_model, N_data]);

    if N_use <= n_arx
        error('estimate_uncertainty: N_use (%d) must exceed n_arx (%d).', ...
            N_use, n_arx);
    end
    if N_use - n_arx < 10
        warning(['estimate_uncertainty: only %d valid ARX residual ' ...
                 'samples; the quantile estimate may be unreliable.'], ...
                 N_use - n_arx);
    end

    %% =====================================================================
    % 3. Estimation parameters
    % =====================================================================
    if isfield(Params, 'safety_factor') && ~isempty(Params.safety_factor)
        safety_factor = Params.safety_factor;
    else
        safety_factor = 1.2;
    end
    if safety_factor < 1
        warning(['estimate_uncertainty: safety_factor = %.2f < 1; ' ...
                 'bounds may be too tight.'], safety_factor);
    end

    if isfield(Params, 'q_level') && ~isempty(Params.q_level)
        q_level = Params.q_level;
    else
        q_level = 0.95;
    end
    if q_level <= 0 || q_level >= 1
        error('estimate_uncertainty: q_level must lie strictly in (0, 1).');
    end

    fprintf('estimate_uncertainty: safety_factor = %.2f, q_level = %.2f\n', ...
        safety_factor, q_level);

    %% =====================================================================
    % 4. ARX innovation residuals (latent standardized-deviation domain)
    % =====================================================================
    eta_all = NaN(N_use, A_lv);

    for a = 1:A_lv
        u_seq = U_lv(1:N_use, a);
        t_seq = T_lv(1:N_use, a);

        theta = Theta{a};
        if isempty(theta)
            error('estimate_uncertainty: empty ARX parameters at LV %d.', a);
        end
        theta = theta(:);
        if length(theta) ~= 2 * n_arx
            error(['estimate_uncertainty: ARX parameter length at LV %d ' ...
                   'is %d, expected %d.'], a, length(theta), 2 * n_arx);
        end
        if any(~isfinite(theta))
            error('estimate_uncertainty: ARX parameters at LV %d contain NaN/Inf.', a);
        end

        alpha = theta(1:n_arx);
        beta  = theta(n_arx + 1 : 2 * n_arx);

        for k = (n_arx + 1):N_use
            u_lag = u_seq(k - 1 : -1 : k - n_arx);
            t_lag = t_seq(k - 1 : -1 : k - n_arx);

            u_pred        = alpha' * u_lag + beta' * t_lag;
            eta_all(k, a) = u_seq(k) - u_pred;
        end
    end

    eta_valid = eta_all(n_arx + 1 : N_use, :);
    eta_valid = eta_valid(all(isfinite(eta_valid), 2), :);

    if isempty(eta_valid)
        error('estimate_uncertainty: no valid ARX residual samples.');
    end

    %% =====================================================================
    % 5. Output reconstruction residuals (standardized deviation form)
    % =====================================================================
    if ~isfield(Model, 'R_y') || isempty(Model.R_y)
        error('estimate_uncertainty: Model.R_y is empty.');
    end

    R_y_model = Model.R_y;

    if size(R_y_model, 1) < N_use
        error(['estimate_uncertainty: Model.R_y has fewer rows (%d) than ' ...
               'N_use (%d).'], size(R_y_model, 1), N_use);
    end
    R_y_model = R_y_model(1:N_use, :);

    if any(any(~isfinite(R_y_model)))
        error('estimate_uncertainty: Model.R_y contains NaN/Inf.');
    end

    if isfield(Model, 'Q') && ~isempty(Model.Q) ...
       && isfield(Identification_data, 'Y0') && ~isempty(Identification_data.Y0)

        Q_pls = Model.Q;
        if size(Q_pls, 2) ~= A_lv
            error('estimate_uncertainty: Model.Q must have A_lv columns.');
        end

        if size(Identification_data.Y0, 1) >= N_use
            U_use     = U_lv(1:N_use, 1:A_lv);
            R_y_check = Identification_data.Y0(1:N_use, :) - U_use * Q_pls';
            rel_diff  = norm(R_y_check - R_y_model, 'fro') ...
                        / max(norm(R_y_model, 'fro'), eps);

            if rel_diff > 1e-3
                warning(['estimate_uncertainty: Model.R_y differs from ' ...
                         'Y0 - U*Q'' (relative Frobenius diff %.3e). ' ...
                         'Using Model.R_y.'], rel_diff);
            end
        end
    end

    r_valid = R_y_model(all(isfinite(R_y_model), 2), :);
    if isempty(r_valid)
        error('estimate_uncertainty: no valid output reconstruction residuals.');
    end

    %% =====================================================================
    % 6. Bounded uncertainties (per-channel quantiles)
    % =====================================================================
    eta_q = column_quantile(abs(eta_valid), q_level)';
    r_q   = column_quantile(abs(r_valid),   q_level)';

    eta_bound = safety_factor * eta_q;
    r_bound   = safety_factor * r_q;

    eta_bound(eta_bound < 1e-8) = 1e-8;
    r_bound(r_bound     < 1e-8) = 1e-8;

    if length(eta_bound) ~= A_lv
        error(['estimate_uncertainty: eta_bound length %d must equal ' ...
               'A_lv %d.'], length(eta_bound), A_lv);
    end
    if length(r_bound) ~= size(R_y_model, 2)
        error(['estimate_uncertainty: r_bound length %d must equal ' ...
               'output dimension %d.'], length(r_bound), size(R_y_model, 2));
    end

    %% =====================================================================
    % 7. Package results
    % =====================================================================
    UncSets.eta_bound     = eta_bound;
    UncSets.r_bound       = r_bound;
    UncSets.eta           = eta_all;
    UncSets.r             = R_y_model;
    UncSets.safety_factor = safety_factor;
    UncSets.q_level       = q_level;
    UncSets.n_eta_samples = size(eta_valid, 1);
    UncSets.n_r_samples   = size(r_valid, 1);

    %% =====================================================================
    % 8. Summary
    % =====================================================================
    fprintf('estimate_uncertainty: bounds estimated (standardized deviation).\n');
    fprintf('  ARX residual dimension : %d\n', length(eta_bound));
    fprintf('  Output residual dim    : %d\n', length(r_bound));
    fprintf('  Valid ARX samples      : %d\n', size(eta_valid, 1));
    fprintf('  Valid output samples   : %d\n', size(r_valid, 1));
    fprintf('  eta_bound              : ');
    fprintf('%.4e ', eta_bound); fprintf('\n');
    fprintf('  r_bound                : ');
    fprintf('%.4e ', r_bound);   fprintf('\n');
end

% =========================================================================
% Local function: column_quantile
% =========================================================================
function q = column_quantile(X, p)
    Xs = sort(X, 1);
    n  = size(Xs, 1);

    if n == 1
        q = Xs;
        return;
    end

    idx = p * (n - 1) + 1;
    lo  = floor(idx);
    hi  = ceil(idx);
    w   = idx - lo;

    if lo == hi
        q = Xs(lo, :);
    else
        q = (1 - w) * Xs(lo, :) + w * Xs(hi, :);
    end
end
