function [t_feasible, u_feasible, y_feasible, v_feasible, x_feasible] = ...
        Initial_Feasible(x_nominal_init, Identification_data, Model, Tube, Params)

    N = Params.Ble;
    ctrl_start = Params.ctrl_start;
    N_ctrl = N - ctrl_start;  % 60 步

    A_lv = Params.A_lv;
    n_in = Params.n_in_opt; % 5
    n_out = Params.n_out;
    nx = length(x_nominal_init);

    barA = Model.barA; barB = Model.barB; barC = Model.barC;
    Q = Model.Q; P_mat = Model.P;
    X_mean = Identification_data.X_mean(:);
    Y_mean = Identification_data.Y_mean(:);
    S_v = diag(Model.X_std); S_y = diag(Model.Y_std);

    n_t = A_lv * N_ctrl;

    [Phi_x, x0_c, Phi_y, y0_c, Phi_v, v0_c] = ...
        build_lifting_matrices(N_ctrl, nx, n_out, n_in, A_lv, barA, barB, barC, ...
            S_y, Q, S_v, P_mat, X_mean, Y_mean, x_nominal_init);

    Aineq = []; bineq = [];
    for k = 0:N_ctrl-1
        k_abs = ctrl_start + k;
        rv = k * n_in + (1:n_in);
        ry = k * n_out + (1:n_out);

        % ★ 维度切片
        h_v_k = [Params.u_phys_max(1:n_in) - Params.u_base(1:n_in, k_abs + 1);
                 -Params.u_phys_min(1:n_in) + Params.u_base(1:n_in, k_abs + 1)];
        h_y_k = [Params.y_phys_max - Params.y_base(:, k_abs + 2);
                 -Params.y_phys_min + Params.y_base(:, k_abs + 2)];

        h_v_tight_k = h_v_k - Tube.delta_v;
        h_y_tight_k = h_y_k - Tube.delta_y;

        Aineq = [Aineq; Tube.H_v * Phi_v(rv, :); Tube.H_y * Phi_y(ry, :)];
        bineq = [bineq; h_v_tight_k - Tube.H_v * v0_c(rv);
                        h_y_tight_k - Tube.H_y * y0_c(ry)];
    end

    R_w = Params.R_weight; if isscalar(R_w), R_w = R_w * eye(A_lv); end
    H_qp = 2 * kron(eye(N_ctrl), R_w); H_qp = (H_qp + H_qp') / 2;
    f_qp = zeros(n_t,1);

    row_max = max(abs(Aineq), [], 2); row_max(row_max < 1e-10) = 1;
    Aineq_scaled = Aineq ./ row_max; bineq_scaled = bineq ./ row_max;

    if exist('OCTAVE_VERSION','builtin')
        [t_opt, ~, info] = qp(zeros(n_t,1), H_qp, f_qp, [], [], ...
            -inf(n_t,1), inf(n_t,1), -inf(size(bineq_scaled)), Aineq_scaled, bineq_scaled);
        if info.info ~= 0, error('Initial_Feasible: 无解！info=%d', info.info); end
    else
        opts = optimoptions('quadprog','Display','off');
        [t_opt, ~, flag] = quadprog(H_qp, f_qp, Aineq, bineq, [], [], [], [], [], opts);
        if flag <= 0, error('Initial_Feasible: 无解！flag=%d', flag); end
    end

    t_vec = t_opt;
    t_feasible = reshape(t_vec, A_lv, N_ctrl);
    x_feasible = [x_nominal_init, reshape(Phi_x * t_vec + x0_c, nx, N_ctrl)];
    u_feasible = zeros(A_lv, N_ctrl);
    for j = 1:N_ctrl, u_feasible(:,j) = barC * x_feasible(:, j+1); end
    y_feasible = reshape(Phi_y * t_vec + y0_c, n_out, N_ctrl);
    v_feasible = reshape(Phi_v * t_vec + v0_c, n_in,  N_ctrl);
end
