function Tube = compute_tube(Identification_data, Model, UncSets, Params)
if exist('OCTAVE_VERSION','builtin'), pkg load control; end

A_bar = Model.barA; B_bar = Model.barB;
C_bar = Model.barC; E_bar = Model.barE;
Q_pls = Model.Q;    P_mat = Model.P;

[nx, ~] = size(A_bar);
[~, nt] = size(B_bar);
A_lv    = Params.A_lv;
n_arx   = Params.n_arx;
n_state = 2*n_arx - 1;
n_in    = Params.n_in_opt; % ★ 使用 n_in_opt

R_u = Model.R_u;
W_r_mat = diag(UncSets.r_bound.^2);

%% 1. LQR
Ql = Params.Q_lqr; Rl = Params.R_lqr;
P_blk   = zeros(nx); K_blk   = zeros(nt, nx); A_K_blk = zeros(nx);

for a = 1:A_lv
    idx = (a-1)*n_state + (1:n_state);
    A_i = A_bar(idx, idx); B_i = B_bar(idx, a); E_i = E_bar(idx, a);

    [K_place, ~, ~] = dlqr(A_i, B_i, Ql*eye(n_state), Rl);
    K_bar_i = -K_place; A_K_i = A_i + B_i * K_bar_i;
    rho = max(abs(eig(A_K_i)));
    if rho >= 1, error('compute_tube: unstable.'); end

    W_eta_i = UncSets.eta_bound(a)^2;
    EWE     = E_i * W_eta_i * E_i'; EWE = 0.5*(EWE + EWE');

    var_eps_u_a = R_u(:, a)' * W_r_mat * R_u(:, a);
    W_eps_x_a = zeros(n_state);
    W_eps_x_a(1:n_arx, 1:n_arx) = var_eps_u_a * eye(n_arx);

    EWE = EWE + B_i * K_bar_i * W_eps_x_a * K_bar_i' * B_i';
    EWE = 0.5*(EWE + EWE');

    Ph = Params.P_horizon; P_i = zeros(n_state); A_pow = eye(n_state);
    for j = 0:Ph-1
        P_i = P_i + A_pow * EWE * A_pow';
        A_pow = A_K_i * A_pow;
    end
    P_i = 0.5*(P_i + P_i');
    if min(eig(P_i)) <= 0
        P_i = P_i + 1e-8 * max(1, norm(P_i,'fro')) * eye(n_state);
    end
    P_blk(idx,idx) = P_i; K_blk(a, idx) = K_bar_i; A_K_blk(idx,idx) = A_K_i;
end
P = P_blk; K = K_blk; A_K = A_K_blk; rho_all = max(abs(eig(A_K)));

%% 2. Input tightening
X_std = Identification_data.X_std(:); % 5x1
S_v = diag(X_std); % 5x5

W_eps_x_total = zeros(nx);
for a = 1:A_lv
    idx = (a-1)*n_state + (1:n_state);
    var_eps_u_a = R_u(:, a)' * W_r_mat * R_u(:, a);
    W_eps_x_a = zeros(n_state);
    W_eps_x_a(1:n_arx, 1:n_arx) = var_eps_u_a * eye(n_arx);
    W_eps_x_total(idx, idx) = W_eps_x_a;
end

W_v = S_v * P_mat * K * (P + W_eps_x_total) * K' * P_mat' * S_v';
W_v = 0.5*(W_v + W_v');
H_v = Params.H_v; % 10x5
n_v = size(H_v,1);

delta_v = zeros(n_v,1);
for i = 1:n_v
    h_i = H_v(i,:);
    q   = h_i * W_v * h_i';
    delta_v(i) = sqrt(max(q, 0));
end

%% 3. Output tightening
Y_std = Identification_data.Y_std(:);
S_y = diag(Y_std);
W_r = diag(UncSets.r_bound.^2);
W_y_r = S_y * W_r * S_y';

W_y_state = S_y * Q_pls * C_bar * P * C_bar' * Q_pls' * S_y';
W_y_state = 0.5*(W_y_state + W_y_state');
H_y = Params.H_y;
n_y = size(H_y,1);

delta_y = zeros(n_y,1);
for i = 1:n_y
    h_i = H_y(i,:);
    qs = h_i * W_y_state * h_i';
    qr = h_i * W_y_r     * h_i';
    delta_y(i) = sqrt(max(qs,0)) + sqrt(max(qr,0));
end

%% 4. Package
Tube.P = P; Tube.K = K; Tube.A_K = A_K; Tube.spectral_radius = rho_all;
Tube.P_horizon = Params.P_horizon;
Tube.W_eta = diag(UncSets.eta_bound.^2);
Tube.W_r   = W_r;
Tube.W_v   = W_v;
Tube.W_eps_x = W_eps_x_total;
Tube.delta_v = delta_v; Tube.delta_y = delta_y;
Tube.H_v = H_v; Tube.H_y = H_y;

fprintf('compute_tube: rho=%.6f, delta_v=[%s], delta_y=[%s]\n', ...
    rho_all, num2str(delta_v','%.3e '), num2str(delta_y','%.3e '));
end
