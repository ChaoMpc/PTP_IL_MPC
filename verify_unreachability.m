function Result = verify_unreachability(Identification_data, Model, Tube, Params)
% 直接求解问题 (55)，验证参考点不可达并给出 y*_s (R1-Major 9)
%   min_{T ∈ C_T}  || M (Y(T) - Y_ref) ||_Q^2

N = Params.Ble; A_lv = Params.A_lv; n_out = Params.n_out; n_in = Params.n_in;
barA = Model.barA; barB = Model.barB; barC = Model.barC;
nx = size(barA,1);

Q_pls = Model.Q;
R_x = Model.R_x; R_x_inv = pinv(R_x');
X_mean = Identification_data.X_mean(:); X_std = Identification_data.X_std(:);
Y_mean = Identification_data.Y_mean(:); Y_std  = Identification_data.Y_std(:);
S_v = diag(X_std); S_y = diag(Y_std);

[Phi_x, ~, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q_pls, S_v, R_x_inv, X_mean, Y_mean, zeros(nx,1));

% 掩码与参考
mask = zeros(n_out, N); ref_traj = zeros(n_out, N);
for i = 1:numel(Params.Kp)
    k_idx = Params.Kp(i); mask(1, k_idx) = 1;
    ref_traj(1, k_idx) = Params.Ref_points(i);
end
Mask_vec = zeros(n_out*(N-1),1); Y_ref_vec = zeros(n_out*(N-1),1);
for i = 1:N-1
    k = i+1; row_idx = (i-1)*n_out+1 : i*n_out;
    Y_ref_vec(row_idx) = ref_traj(:, k);
    Mask_vec(row_idx)  = mask(:, k)';
end
Mask_mat = diag(Mask_vec);
Q_blk = kron(eye(N-1), Params.Q_weight);

H_qp = Phi_y' * Mask_mat * Q_blk * Mask_mat * Phi_y;
H_qp = 0.5*(H_qp + H_qp');
H_qp = H_qp + 1e-10*eye(A_lv*(N-1));
f_qp = Phi_y' * Mask_mat * Q_blk * Mask_mat * (y0_const - Y_ref_vec);

H_v_blk = kron(eye(N-1), Params.H_v);
H_y_blk = kron(eye(N-1), Params.H_y);
A_ineq = [H_v_blk * Phi_v; H_y_blk * Phi_y];
b_ineq = [repmat(Tube.h_v_tight, N-1,1) - H_v_blk*v0_const;
          repmat(Tube.h_y_tight, N-1,1) - H_y_blk*y0_const];

t_vec_opt = qp(zeros(A_lv*(N-1),1), H_qp, f_qp, [], [], ...
    -inf(A_lv*(N-1),1), inf(A_lv*(N-1),1), ...
    -inf(size(A_ineq,1),1), A_ineq, b_ineq);

Y_stack = Phi_y * t_vec_opt + y0_const;
Y_nom   = reshape(Y_stack, n_out, N-1);
V_nom   = reshape(Phi_v * t_vec_opt + v0_const, n_in, N-1);

Result.t_vec_opt = t_vec_opt;
Result.Y_nom     = Y_nom;
Result.V_nom     = V_nom;
Result.J_opt     = 0.5*t_vec_opt'*H_qp*t_vec_opt + f_qp'*t_vec_opt;

% 提取 Kp 处 y*_s
Result.y_s_at_Kp = zeros(numel(Params.Kp), n_out);
Result.nom_error_at_Kp = zeros(numel(Params.Kp), 1);
for i = 1:numel(Params.Kp)
    k_idx = Params.Kp(i);
    if k_idx > 1
        Result.y_s_at_Kp(i, :) = Y_nom(:, k_idx-1)';
        Result.nom_error_at_Kp(i) = Y_nom(1, k_idx-1) - Params.Ref_points(i);
    end
end

% 活跃约束
tol = 1e-6;
viol = A_ineq * t_vec_opt - b_ineq;
active_idx = find(viol > -tol);
Result.active = active_idx;
Result.n_active = numel(active_idx);
end
