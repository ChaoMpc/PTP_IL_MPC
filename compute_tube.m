function Tube = compute_tube(Identification_data, Model, UncSets, Params)
    % ========================================================================
    % compute_tube -- Compute the robust invariant tube via finite-horizon
    % propagation.
    %
    % For each latent channel, build a state-feedback gain K_bar that keeps
    % the state error e = x - x_tilde inside an ellipsoidal invariant set,
    % then compute the input/output constraint tightening margins delta_v,
    % delta_y for use by the ILC-MPC.
    %
    % ------------------------------ Model convention ----------------------
    %   Latent dynamics (block diagonal):
    %       x(k+1) = barA*x(k) + barB*t(k) + barE*eta(k)
    %       u(k)   = barC*x(k)
    %   Tube feedback:
    %       t_actual(k) = t_nominal(k) + K_bar*(x_actual(k) - x_nominal(k))
    %   Error dynamics:
    %       e(k+1) = (barA + barB*K_bar) * e(k) + barE*eta(k) = A_K*e(k) + barE*eta(k)
    %
    % ------------------------------ Error ellipsoid -----------------------
    %   Finite-horizon covariance approximation:
    %       P = sum_{j=0}^{P_horizon-1} A_K^j * barE * W_eta * barE' * (A_K^j)'
    %   with W_eta = diag(eta_bound.^2).
    %
    % ------------------------------ Constraint tightening -----------------
    %   Input:  W_v = S_v * R_x_inv * K_bar * P * K_bar' * R_x_inv' * S_v'
    %           delta_v(i) = sqrt( H_v(i,:) * W_v * H_v(i,:)' )
    %   Output: W_y_state = S_y * Q * barC * P * barC' * Q' * S_y'
    %           W_y_r     = S_y * W_r * S_y'
    %           delta_y(i) = sqrt( H_y(i,:) * W_y_state * H_y(i,:)' )
    %                      + sqrt( H_y(i,:) * W_y_r     * H_y(i,:)' )
    %   (Support function of Minkowski sum = sum of support functions.)
    %
    % ------------------------------ Inputs --------------------------------
    %   Identification_data : X_mean, X_std, Y_mean, Y_std
    %   Model               : barA, barB, barC, barE, R_x, Q
    %   UncSets             : eta_bound (A_lv x 1), r_bound (n_y x 1)
    %   Params              : A_lv, n_arx, Ble (or P_horizon), H_v, h_v,
    %                         H_y, h_y, pole_locations
    %
    % ------------------------------ Outputs -------------------------------
    %   Tube.P, Tube.K, Tube.A_K, Tube.delta_v, Tube.delta_y,
    %   Tube.h_v_tight, Tube.h_y_tight, Tube.W_eta, Tube.W_r, Tube.W_v,
    %   Tube.spectral_radius
    % ========================================================================

    %% ==================== 1. Read model matrices ====================
    A_bar = Model.barA;
    B_bar = Model.barB;
    C_bar = Model.barC;
    E_bar = Model.barE;
    R_x = Model.R_x;
    Q_pls = Model.Q;

    [nx, ~] = size(A_bar);
    [~, nt] = size(B_bar);
    [ny, ~] = size(C_bar);

    %% ==================== 2. Consistency checks ====================
    if size(B_bar, 1) ~= nx || size(C_bar, 2) ~= nx || size(E_bar, 1) ~= nx
        error('compute_tube: inconsistent Model matrix dimensions.');
    end

    if size(R_x, 2) ~= nt
        error('compute_tube: R_x must have A_lv columns.');
    end

    if size(Q_pls, 2) ~= nt
        error('compute_tube: Q must have A_lv columns.');
    end

    if ~isfield(Params, 'A_lv') || isempty(Params.A_lv)
        error('compute_tube: Params.A_lv is required.');
    end

    A_lv = Params.A_lv;

    if A_lv ~= nt
        error('compute_tube: Params.A_lv (%d) differs from B_bar columns (%d).', A_lv, nt);
    end

    if ~isfield(Params, 'n_arx') || isempty(Params.n_arx)
        error('compute_tube: Params.n_arx is required.');
    end

    n_arx = Params.n_arx;

    n_state = 2 * n_arx - 1;

    if nx ~= A_lv * n_state
        error('compute_tube: total state dim %d != A_lv*(%d) = %d.', ...
            nx, n_state, A_lv * n_state);
    end

    %% ==================== 3. Prediction horizon ====================
    if isfield(Params, 'P_horizon') &&~isempty(Params.P_horizon)
        P_horizon = Params.P_horizon;
    elseif isfield(Params, 'Ble') &&~isempty(Params.Ble)
        P_horizon = Params.Ble;
    else
        error('compute_tube: Params.P_horizon or Params.Ble is required.');
    end

    if P_horizon < 1
        error('compute_tube: prediction horizon must be >= 1.');
    end

    %% ==================== 4. Uncertainty bounds ====================
    if ~isfield(UncSets, 'eta_bound') || isempty(UncSets.eta_bound)
        error('compute_tube: UncSets.eta_bound is missing.');
    end

    if ~isfield(UncSets, 'r_bound') || isempty(UncSets.r_bound)
        error('compute_tube: UncSets.r_bound is missing.');
    end

    eta_bound = UncSets.eta_bound(:);
    r_bound = UncSets.r_bound(:);
    fprintf('compute_tube received eta_bound = ');
    fprintf('%.4e ', eta_bound);
    fprintf('\n');

    if length(eta_bound) ~= A_lv
        error('compute_tube: eta_bound length (%d) must equal A_lv (%d).', ...
            length(eta_bound), A_lv);
    end

    if length(r_bound) ~= ny
        error('compute_tube: r_bound length (%d) must equal n_y (%d).', ...
            length(r_bound), ny);
    end

    if any(eta_bound < 0) || any(r_bound < 0)
        error('compute_tube: uncertainty bounds must be nonnegative.');
    end

    %% ==================== 5. Normalization scales ====================
    if ~isfield(Identification_data, 'X_std') ||~isfield(Identification_data, 'Y_std')
        error('compute_tube: Identification_data must contain X_std and Y_std.');
    end

    S_v = diag(Identification_data.X_std);
    S_y = diag(Identification_data.Y_std);

    if size(S_v, 1) ~= nt || size(S_y, 1) ~= ny
        error('compute_tube: normalization matrix dimensions mismatch.');
    end

    %% ==================== 6. Constraint matrices ====================
    if ~isfield(Params, 'H_v') ||~isfield(Params, 'h_v') || ...
            ~isfield(Params, 'H_y') ||~isfield(Params, 'h_y')
        error('compute_tube: Params must contain H_v, h_v, H_y, h_y.');
    end

    H_v = Params.H_v;
    h_v = Params.h_v(:);
    H_y = Params.H_y;
    h_y = Params.h_y(:);

    n_v_constr = size(H_v, 1);
    n_y_constr = size(H_y, 1);

    %% ==================== 7. Per-channel gain and ellipsoid ====================
    P_blk = zeros(nx, nx);
    K_blk = zeros(nt, nx);
    A_K_blk = zeros(nx, nx);
    eig_closed_all = zeros(nx, 1);

    if isfield(Params, 'pole_locations') &&~isempty(Params.pole_locations)
        desired_poles = Params.pole_locations(:);

        if length(desired_poles) ~= n_state
            warning('compute_tube: pole_locations length (%d) differs from n_state (%d); using default.', ...
                length(desired_poles), n_state);
            desired_poles = linspace(0.5, 0.8, n_state).';
        end

    else
        desired_poles = linspace(0.5, 0.8, n_state).';
    end

    if any(abs(desired_poles) >= 1)
        error('compute_tube: all desired poles must be strictly inside the unit circle.');
    end

    for a = 1:A_lv
        idx = (a - 1) * n_state + 1:a * n_state;
        A_i = A_bar(idx, idx);
        B_i = B_bar(idx, a);
        E_i = E_bar(idx, a);

        % Pole placement with LQR fallback
        if rank(ctrb(A_i, B_i)) < n_state
            warning('compute_tube: LV %d is not fully controllable; using LQR.', a);
        end

        % Use LQR instead of place. place generates huge gains when the desired
        % poles are far from the plant poles, which inflates delta_v.
        % Tuning R controls the gain magnitude.
        Q_lqr = Params.Q_lqr * eye(n_state);
        R_lqr = Params.R_lqr;
        [K_place, ~, ~] = dlqr(A_i, B_i, Q_lqr, R_lqr);

        % Sign convention: K_bar_i = -K_place so that A_K_i = A_i + B_i*K_bar_i
        % is Schur stable, matching t_actual = t_nominal + K_bar*e.
        K_bar_i = -K_place;
        A_K_i = A_i + B_i * K_bar_i;

        rho_i = max(abs(eig(A_K_i)));

        if rho_i >= 1
            error('compute_tube: LV %d closed-loop not Schur stable (rho=%.4f).', a, rho_i);
        end

        % Finite-horizon error covariance
        W_eta_i = eta_bound(a)^2;
        EWE = E_i * W_eta_i * E_i';
        EWE = 0.5 * (EWE + EWE');

        P_i = zeros(n_state, n_state);
        A_pow = eye(n_state);

        for j = 0:P_horizon - 1
            P_i = P_i + A_pow * EWE * A_pow';
            A_pow = A_K_i * A_pow;
        end

        P_i = 0.5 * (P_i + P_i');

        if min(eig(P_i)) <= 0
            P_i = P_i + 1e-8 * max(1, norm(P_i, 'fro')) * eye(n_state);
        end

        P_blk(idx, idx) = P_i;
        K_blk(a, idx) = K_bar_i;
        A_K_blk(idx, idx) = A_K_i;
        eig_closed_all(idx) = eig(A_K_i);
    end

    P = P_blk;
    K_bar = K_blk;
    A_K = A_K_blk;
    spectral_radius = max(abs(eig_closed_all));

    fprintf('compute_tube: finite-horizon tube computed.\n');
    fprintf('  Total state dim      : %d\n', nx);
    fprintf('  Per-channel state dim: %d\n', n_state);
    fprintf('  Number of LVs        : %d\n', A_lv);
    fprintf('  Prediction horizon   : %d\n', P_horizon);
    fprintf('  Closed-loop radius   : %.6f\n', spectral_radius);

    %% ==================== 8. Input tightening ====================
    if cond(R_x) > 1e8
        warning('compute_tube: R_x is ill-conditioned (%.2e).', cond(R_x));
    end

    R_x_inv = pinv(R_x');

    W_v = S_v * R_x_inv * K_bar * P * K_bar' * R_x_inv' * S_v';
    W_v = 0.5 * (W_v + W_v');

    delta_v = zeros(n_v_constr, 1);

    for i = 1:n_v_constr
        h_i = H_v(i, :);
        quad = h_i * W_v * h_i';
        if quad < 0, quad = 0; end
        delta_v(i) = sqrt(quad);
    end

    %% ==================== 9. Output tightening ====================
    W_r = diag(r_bound.^2);
    W_y_state = S_y * Q_pls * C_bar * P * C_bar' * Q_pls' * S_y';
    W_y_r = S_y * W_r * S_y';
    W_y_state = 0.5 * (W_y_state + W_y_state');
    W_y_r = 0.5 * (W_y_r + W_y_r');

    delta_y = zeros(n_y_constr, 1);

    for i = 1:n_y_constr
        h_i = H_y(i, :);
        qs = h_i * W_y_state * h_i';
        qr = h_i * W_y_r * h_i';
        if qs < 0, qs = 0; end
        if qr < 0, qr = 0; end
        delta_y(i) = sqrt(qs) + sqrt(qr);
    end

    %% ==================== 10. Tightened bounds ====================
    h_v_tight = h_v - delta_v;
    h_y_tight = h_y - delta_y;

    if any(h_v_tight < 0)
        warning('compute_tube: input tightened bounds become negative; MPC may be infeasible.');
    end

    if any(h_y_tight < 0)
        warning('compute_tube: output tightened bounds become negative; MPC may be infeasible.');
    end

    %% ==================== 11. Package ====================
    Tube.P = P;
    Tube.K = K_bar;
    Tube.A_K = A_K;
    Tube.desired_poles = desired_poles;
    Tube.eig_closed = eig_closed_all;
    Tube.spectral_radius = spectral_radius;
    Tube.P_horizon = P_horizon;

    Tube.W_eta = diag(eta_bound.^2);
    Tube.W_r = W_r;
    Tube.W_v = W_v;

    Tube.delta_v = delta_v;
    Tube.delta_y = delta_y;
    Tube.h_v_tight = h_v_tight;
    Tube.h_y_tight = h_y_tight;

    Tube.H_v = H_v;
    Tube.H_y = H_y;
    Tube.h_v = h_v;
    Tube.h_y = h_y;

    %% ==================== 12. Summary ====================
    fprintf('  delta_v : ');
    fprintf('%.4e ', delta_v);
    fprintf('\n');
    fprintf('  delta_y : ');
    fprintf('%.4e ', delta_y);
    fprintf('\n');
    fprintf('  h_v_tight (first 4) : ');
    fprintf('%.4e ', h_v_tight(1:min(4, end)));
    fprintf('\n');
    fprintf('  h_y_tight (first 4) : ');
    fprintf('%.4e ', h_y_tight(1:min(4, end)));
    fprintf('\n');
    fprintf('compute_tube: completed successfully.\n');
end
