function [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params)
% =========================================================================
% Initial_Feasible -- 生成初始可行名义轨迹（第 0 批）。
%
% 作用：
%   通过纯 QP 求解，找到一个满足所有 tightened 约束的初始潜输入序列。
%   这保证了批次 1 的 IL-MPC 优化问题有可行的起点（满足 Assumption 4）。
% =========================================================================

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
