function Tube = compute_tube(Identification_data, Model, UncSets, Params)
% =========================================================================
% compute_tube -- 计算鲁棒不变 tube 与约束紧化裕度。
%
% 比例约定
% --------
% 本函数产出 PHYSICAL DEVIATION 空间的裕度与 tightened 界：
%       v(k) = u(k) - u_ss      [lb/min]
%       y(k) = y(k) - y_ss      [wt%]
%
% 依照审稿人意见补充：
%   - 完整报告 W_eta, W_r, P, alpha, delta_v, delta_y (R1-Major 4)
%   - tightened 集合非空性检查 (R1-Major 6)
% =========================================================================

if exist('OCTAVE_VERSION', 'builtin')
    pkg load control;
end

%% 1. 模型矩阵
A_bar = Model.barA;
B_bar = Model.barB;
C_bar = Model.barC;
E_bar = Model.barE;
R_x   = Model.R_x;
Q_pls = Model.Q;

[nx, ~] = size(A_bar);
[~, nt] = size(B_bar);
[ny, ~] = size(C_bar);

%% 2. 一致性检查
if size(B_bar,1) ~= nx || size(C_bar,2) ~= nx || size(E_bar,1) ~= nx
    error('compute_tube: inconsistent Model matrix dimensions.');
end
if ~isfield(Params,'A_lv') || isempty(Params.A_lv)
    error('compute_tube: Params.A_lv required.');
end
A_lv = Params.A_lv;
if A_lv ~= nt
    error('compute_tube: A_lv (%d) != B_bar columns (%d).', A_lv, nt);
end
if size(R_x,2) ~= A_lv
    error('compute_tube: R_x must have A_lv columns.');
end
if size(Q_pls,2) ~= A_lv
    error('compute_tube: Q must have A_lv columns.');
end
if size(Q_pls,1) ~= ny
    error('compute_tube: Q rows (%d) != C_bar rows (%d).', size(Q_pls,1), ny);
end
if ~isfield(Params,'n_arx') || isempty(Params.n_arx)
    error('compute_tube: Params.n_arx required.');
end
n_arx   = Params.n_arx;
n_state = 2*n_arx - 1;
if nx ~= A_lv*n_state
    error('compute_tube: total state dim %d != A_lv*(%d) = %d.', ...
        nx, n_state, A_lv*n_state);
end

%% 3. 预测时域
if isfield(Params,'P_horizon') && ~isempty(Params.P_horizon)
    P_horizon = Params.P_horizon;
elseif isfield(Params,'Ble') && ~isempty(Params.Ble)
    P_horizon = Params.Ble;
else
    error('compute_tube: Params.P_horizon or Params.Ble required.');
end
if P_horizon < 1, error('compute_tube: P_horizon must be >= 1.'); end

%% 4. 不确定界
if ~isfield(UncSets,'eta_bound') || isempty(UncSets.eta_bound)
    error('compute_tube: UncSets.eta_bound missing.');
end
if ~isfield(UncSets,'r_bound') || isempty(UncSets.r_bound)
    error('compute_tube: UncSets.r_bound missing.');
end
eta_bound = UncSets.eta_bound(:);
r_bound   = UncSets.r_bound(:);
if length(eta_bound) ~= A_lv
    error('compute_tube: eta_bound length (%d) != A_lv (%d).', ...
        length(eta_bound), A_lv);
end
if length(r_bound) ~= ny
    error('compute_tube: r_bound length (%d) != n_y (%d).', ...
        length(r_bound), ny);
end
if any(eta_bound<0) || any(r_bound<0)
    error('compute_tube: uncertainty bounds must be nonnegative.');
end

%% 5. 标准化尺度 (仅内部使用)
if ~isfield(Identification_data,'X_std') || ~isfield(Identification_data,'Y_std')
    error('compute_tube: X_std / Y_std required.');
end
X_std = Identification_data.X_std(:);
Y_std = Identification_data.Y_std(:);
S_v   = diag(X_std);
S_y   = diag(Y_std);
if size(S_v,1) ~= nt || size(S_y,1) ~= ny
    error('compute_tube: normalization matrix dimensions mismatch.');
end

%% 6. 约束 (物理偏差)
if ~isfield(Params,'h_v_phys') || isempty(Params.h_v_phys) ...
|| ~isfield(Params,'h_y_phys') || isempty(Params.h_y_phys)
    error('compute_tube: Params.h_v_phys / h_y_phys required.');
end
if ~isfield(Params,'H_v') || ~isfield(Params,'H_y')
    error('compute_tube: Params.H_v / H_y required.');
end
H_v = Params.H_v;
H_y = Params.H_y;
h_v_phys = Params.h_v_phys(:);
h_y_phys = Params.h_y_phys(:);
n_v_constr = size(H_v,1);
n_y_constr = size(H_y,1);
if length(h_v_phys) ~= n_v_constr
    error('compute_tube: length(h_v_phys)=%d != size(H_v,1)=%d.', ...
        length(h_v_phys), n_v_constr);
end
if length(h_y_phys) ~= n_y_constr
    error('compute_tube: length(h_y_phys)=%d != size(H_y,1)=%d.', ...
        length(h_y_phys), n_y_constr);
end
if size(H_v,2) ~= nt
    error('compute_tube: H_v columns (%d) != A_lv (%d).', size(H_v,2), nt);
end
if size(H_y,2) ~= ny
    error('compute_tube: H_y columns (%d) != n_y (%d).', size(H_y,2), ny);
end

%% 7. 逐通道 LQR 与误差椭球
if isfield(Params,'Q_lqr') && ~isempty(Params.Q_lqr), Q_lqr_val = Params.Q_lqr; else, Q_lqr_val = 10; end
if isfield(Params,'R_lqr') && ~isempty(Params.R_lqr), R_lqr_val = Params.R_lqr; else, R_lqr_val = 1;  end
if Q_lqr_val <= 0 || R_lqr_val <= 0
    error('compute_tube: Q_lqr and R_lqr must be positive.');
end
if isfield(Params,'verbose_tube') && ~isempty(Params.verbose_tube)
    verbose_tube = logical(Params.verbose_tube);
else
    verbose_tube = true;
end

P_blk       = zeros(nx, nx);
K_blk       = zeros(nt, nx);
A_K_blk     = zeros(nx, nx);
eig_closed_all = zeros(nx, 1);

for a = 1:A_lv
    idx = (a-1)*n_state+1 : a*n_state;
    A_i = A_bar(idx, idx);
    B_i = B_bar(idx, a);
    E_i = E_bar(idx, a);

    if rank(ctrb(A_i, B_i)) < n_state
        warning('compute_tube: LV %d not fully controllable.', a);
    end

    Q_lqr = Q_lqr_val * eye(n_state);
    R_lqr = R_lqr_val;
    [K_place, ~, ~] = dlqr(A_i, B_i, Q_lqr, R_lqr);

    K_bar_i = -K_place;
    A_K_i   = A_i + B_i * K_bar_i;
    rho_i   = max(abs(eig(A_K_i)));
    if rho_i >= 1
        error('compute_tube: LV %d closed-loop not Schur stable (rho=%.4f).', a, rho_i);
    end

    W_eta_i = eta_bound(a)^2;
    EWE     = E_i * W_eta_i * E_i';
    EWE     = 0.5 * (EWE + EWE');

    % 有限时域 RPI 近似：P_i = Σ_{j=0}^{P-1} A_K^j E W_eta E' (A_K')^j
    P_i   = zeros(n_state, n_state);
    A_pow = eye(n_state);
    for j = 0:P_horizon-1
        P_i   = P_i + A_pow * EWE * A_pow';
        A_pow = A_K_i * A_pow;
    end
    P_i = 0.5 * (P_i + P_i');

    if verbose_tube
        fprintf(['  LV %d: eta=%.4e, rho=%.4e, max|K|=%.4e, ' ...
                 'P(1,1)=%.4e\n'], a, eta_bound(a), rho_i, ...
                 max(abs(K_bar_i(:))), P_i(1,1));
    end

    if min(eig(P_i)) <= 0
        P_i = P_i + 1e-8 * max(1, norm(P_i,'fro')) * eye(n_state);
    end

    P_blk(idx, idx)     = P_i;
    K_blk(a, idx)       = K_bar_i;
    A_K_blk(idx, idx)   = A_K_i;
    eig_closed_all(idx) = eig(A_K_i);
end

P               = P_blk;
K_bar           = K_blk;
A_K             = A_K_blk;
spectral_radius = max(abs(eig_closed_all));

fprintf('compute_tube: finite-horizon tube computed.\n');
fprintf('  Total state dim      : %d\n', nx);
fprintf('  Per-channel state dim: %d\n', n_state);
fprintf('  Number of LVs        : %d\n', A_lv);
fprintf('  Prediction horizon   : %d\n', P_horizon);
fprintf('  Closed-loop radius   : %.6f\n', spectral_radius);

%% 8. 输入紧化裕度 (PHYSICAL DEVIATION)
if cond(R_x) > 1e8
    warning('compute_tube: R_x is ill-conditioned (%.2e).', cond(R_x));
end
R_x_inv = pinv(R_x');

W_v = S_v * R_x_inv * K_bar * P * K_bar' * R_x_inv' * S_v';
W_v = 0.5 * (W_v + W_v');

delta_v = zeros(n_v_constr, 1);
for i = 1:n_v_constr
    h_i  = H_v(i,:);
    quad = h_i * W_v * h_i';
    if quad < 0, quad = 0; end
    delta_v(i) = sqrt(quad);
end

%% 9. 输出紧化裕度 (PHYSICAL DEVIATION)
W_r   = diag(r_bound.^2);
W_y_r = S_y * W_r * S_y';
W_y_r = 0.5 * (W_y_r + W_y_r');

if isfield(Params,'use_state_tube') && ~isempty(Params.use_state_tube)
    use_state_tube = logical(Params.use_state_tube);
else
    use_state_tube = true;
end

if use_state_tube
    W_y_state = S_y * Q_pls * C_bar * P * C_bar' * Q_pls' * S_y';
    W_y_state = 0.5 * (W_y_state + W_y_state');
else
    W_y_state = zeros(ny, ny);
end

delta_y = zeros(n_y_constr, 1);
for i = 1:n_y_constr
    h_i = H_y(i,:);
    qs  = h_i * W_y_state * h_i';
    qr  = h_i * W_y_r     * h_i';
    if qs < 0, qs = 0; end
    if qr < 0, qr = 0; end
    delta_y(i) = sqrt(qs) + sqrt(qr);
end

if verbose_tube
    fprintf('\n--- delta_y decomposition ---\n');
    for i = 1:n_y_constr
        h_i = H_y(i,:);
        qs  = sqrt(max(h_i * W_y_state * h_i', 0));
        qr  = sqrt(max(h_i * W_y_r     * h_i', 0));
        fprintf('  constraint %d: state=%.4e, residual=%.4e, total=%.4e\n', ...
            i, qs, qr, qs + qr);
    end
    fprintf('  delta_v        : '); fprintf('%.4e ', delta_v); fprintf('\n');
    fprintf('  delta_y        : '); fprintf('%.4e ', delta_y); fprintf('\n');
end

%% 10. 紧化边界 (PHYSICAL DEVIATION)
h_v_tight = h_v_phys - delta_v;
h_y_tight = h_y_phys - delta_y;

if any(h_v_tight < 0)
    warning(['compute_tube: input tightened bounds become negative. ' ...
             'Enlarge u_phys range or reduce safety_factor.']);
end
if any(h_y_tight < 0)
    warning(['compute_tube: output tightened bounds become negative. ' ...
             'Enlarge y_phys range or reduce safety_factor.']);
end

%% 11. 打包
Tube.P               = P;
Tube.K               = K_bar;
Tube.A_K             = A_K;
Tube.eig_closed      = eig_closed_all;
Tube.spectral_radius = spectral_radius;
Tube.P_horizon       = P_horizon;

Tube.W_eta = diag(eta_bound.^2);
Tube.W_r   = W_r;
Tube.W_v   = W_v;

Tube.delta_v   = delta_v;
Tube.delta_y   = delta_y;
Tube.h_v_tight = h_v_tight;
Tube.h_y_tight = h_y_tight;
Tube.h_v_phys  = h_v_phys;
Tube.h_y_phys  = h_y_phys;

Tube.H_v = H_v;
Tube.H_y = H_y;

%% 12. 完整参数报告 (R1-Major 4)
fprintf('\n--- Uncertainty set parameters (reported to reviewers) ---\n');
fprintf('  eta_bound (per LV) : [%s]\n', num2str(eta_bound', '%.4e '));
fprintf('  r_bound   (per out): [%s]\n', num2str(r_bound',   '%.4e '));
fprintf('  W_eta (diag, std)  : [%s]\n', num2str(diag(Tube.W_eta)', '%.4e '));
fprintf('  W_r   (diag, std)  : [%s]\n', num2str(diag(Tube.W_r)',   '%.4e '));
fprintf('  P: min_eig=%.4e, max_eig=%.4e, cond=%.2e\n', ...
    min(eig(P)), max(eig(P)), cond(P));
if isfield(Params,'tube_alpha') && ~isempty(Params.tube_alpha)
    fprintf('  alpha (tuning)     : %.4f\n', Params.tube_alpha);
else
    fprintf('  alpha (tuning)     : (not provided; ignored in finite-sum approximation)\n');
end
fprintf('  delta_v = [%s]\n', num2str(delta_v', '%.4e '));
fprintf('  delta_y = [%s]\n', num2str(delta_y', '%.4e '));
fprintf('  h_v_phys   [%s]  ->  h_v_tight [%s]\n', ...
    num2str(h_v_phys', '%.4e '), num2str(h_v_tight', '%.4e '));
fprintf('  h_y_phys   [%s]  ->  h_y_tight [%s]\n', ...
    num2str(h_y_phys', '%.4e '), num2str(h_y_tight', '%.4e '));

%% 13. tightened 集合非空性检查 (R1-Major 6)
% 对于盒子约束 H_v = [I; -I], 非空 iff 每分量 tightened 上界 >= 0。
% 对一般多面体仅为经验判据，用以提示用户排查保守性。
nonempty_v = all(h_v_tight(:) > 0);
nonempty_y = all(h_y_tight(:) > 0);
if nonempty_v && nonempty_y
    fprintf('  Tightened set non-emptiness: PASS (all RHS > 0)\n');
else
    fprintf('  Tightened set non-emptiness: WARN (some RHS <= 0)\n');
    fprintf('    input  min(h_v_tight) = %+.4e\n', min(h_v_tight(:)));
    fprintf('    output min(h_y_tight) = %+.4e\n', min(h_y_tight(:)));
end

fprintf('compute_tube: completed successfully.\n');
end
