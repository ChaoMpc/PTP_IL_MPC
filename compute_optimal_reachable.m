function [y_opt_track, y_opt_phys] = compute_optimal_reachable(Identification_data, Model, Tube, Params)
    N = Params.Ble;
    ctrl_start = Params.ctrl_start;
    N_ctrl = N - ctrl_start;
    A_lv = Params.A_lv;
    n_in = Params.n_in_opt; % 5
    n_out = Params.n_out;

    barA = Model.barA; barB = Model.barB; barC = Model.barC;
    Q = Model.Q; P_mat = Model.P;
    X_mean = Identification_data.X_mean(:);
    Y_mean = Identification_data.Y_mean(:);
    S_v = diag(Model.X_std); S_y = diag(Model.Y_std);

    n_t = A_lv * N_ctrl;
    nx = size(barA, 1);
    x0_ctrl = zeros(nx, 1);

    [~, ~, Phi_y, y0_const, Phi_v, v0_const] = ...
        build_lifting_matrices(N_ctrl, nx, n_out, n_in, A_lv, barA, barB, barC, ...
            S_y, Q, S_v, P_mat, X_mean, Y_mean, x0_ctrl);

    n_kp = length(Params.Kp);
    Phi_y_track = zeros(n_kp, n_t);
    y_dev_target = zeros(n_kp, 1);
    y0_const_track = zeros(n_kp, 1);

    for j = 1:n_kp
        k_idx = Params.Kp(j);
        k_rel = k_idx - ctrl_start;
        ry = (k_rel - 1) * n_out + Params.track_idx;

        Phi_y_track(j, :) = Phi_y(ry, :);
        y_dev_target(j) = Params.Ref_points(j);
        y0_const_track(j) = y0_const(ry);
    end

    H_qp = 2 * (Phi_y_track' * Phi_y_track) + 1e-6 * eye(n_t);
    f_qp = 2 * Phi_y_track' * (y0_const_track - y_dev_target);

    Aineq = []; bineq = [];
    for k = 1:N_ctrl
        k_abs = ctrl_start + k;
        rv = (k-1) * n_in + (1:n_in);
        ry = (k-1) * n_out + (1:n_out);

        % ★ 维度切片
        h_v_k = [Params.u_phys_max(1:n_in) - Params.u_base(1:n_in, k_abs);
                 -Params.u_phys_min(1:n_in) + Params.u_base(1:n_in, k_abs)];
        h_y_k = [Params.y_phys_max - Params.y_base(:, k_abs + 1);
                 -Params.y_phys_min + Params.y_base(:, k_abs + 1)];

        h_v_tight_k = h_v_k - Tube.delta_v;
        h_y_tight_k = h_y_k - Tube.delta_y;

        Aineq = [Aineq; Tube.H_v * Phi_v(rv, :); Tube.H_y * Phi_y(ry, :)];
        bineq = [bineq; h_v_tight_k - Tube.H_v * v0_const(rv);
                        h_y_tight_k - Tube.H_y * y0_const(ry)];
    end

    row_max = max(abs(Aineq), [], 2);
    row_max(row_max < 1e-10) = 1;
    Aineq_scaled = Aineq ./ row_max;
    bineq_scaled = bineq ./ row_max;

    if exist('OCTAVE_VERSION', 'builtin')
        [t_opt, ~, info] = qp(zeros(n_t, 1), H_qp, f_qp, [], [], ...
            -inf(n_t, 1), inf(n_t, 1), -inf(size(bineq_scaled)), ...
            Aineq_scaled, bineq_scaled);
        if info.info ~= 0, warning('compute_optimal_reachable: QP 无解！'); t_opt = zeros(n_t, 1); end
    else
        opts = optimoptions('quadprog', 'Display', 'off');
        [t_opt, ~, flag] = quadprog(H_qp, f_qp, Aineq_scaled, bineq_scaled, [], [], [], [], [], opts);
        if flag <= 0, warning('compute_optimal_reachable: QP 无解！'); t_opt = zeros(n_t, 1); end
    end

    y_opt_track = zeros(n_kp, 1);
    for j = 1:n_kp
        k_idx = Params.Kp(j);
        k_rel = k_idx - ctrl_start;
        ry = (k_rel - 1) * n_out + Params.track_idx;
        y_opt_dev = Phi_y(ry, :) * t_opt + y0_const(ry);
        y_opt_track(j) = Params.y_base(Params.track_idx, k_idx + 1) + y_opt_dev;
    end

    y_opt_full_dev = Phi_y * t_opt + y0_const;
    y_opt_phys = reshape(y_opt_full_dev, n_out, N_ctrl) + Params.y_base(:, ctrl_start+1:N);
end
