function Model = build_dypls(Identification_data, Params)
% =========================================================================
% build_dypls -- Dynamic PLS-ARX Model Identification
%                (仅使用控制段数据训练)
% =========================================================================

%% 1. Unpack Data
X0 = Identification_data.X0;  % N x n_in (标准化偏差)
Y0 = Identification_data.Y0;  % N x n_out (标准化偏差)
N = size(X0, 1);
n_in = size(X0, 2);
n_out = size(Y0, 2);

A_lv = Params.A_lv;
n_arx = Params.n_arx;
ctrl_start = Params.ctrl_start;
Ble = Params.Ble;
N_batches = N / Ble;

if mod(N, Ble) ~= 0
    error('build_dypls: 数据长度 N 必须是 Ble 的整数倍');
end

%% 2. Static PLS via NIPALS Algorithm
fprintf('build_dypls: Extracting %d latent variables via NIPALS...\n', A_lv);

T = zeros(N, A_lv);  % Input scores
U = zeros(N, A_lv);  % Output scores
W = zeros(n_in, A_lv);   % Input weights
C = zeros(n_out, A_lv);  % Output weights
P = zeros(n_in, A_lv);   % Input loadings
Q = zeros(n_out, A_lv);  % Output loadings

X_res = X0;
Y_res = Y0;

for a = 1:A_lv
    % NIPALS iteration
    u = Y_res(:, 1);  % 初始化

    for iter = 1:100
        w = X_res' * u;
        w = w / norm(w);  % 归一化

        t = X_res * w;

        c = Y_res' * t;
        c = c / norm(c);

        u_new = Y_res * c;

        if norm(u_new - u) < 1e-8
            break;
        end
        u = u_new;
    end

    % 存储权重和分数
    T(:, a) = t;
    U(:, a) = u;
    W(:, a) = w;
    C(:, a) = c;

    % 计算载荷
    P(:, a) = (X_res' * t) / (t' * t);
    Q(:, a) = (Y_res' * u) / (u' * u);

    % Deflation (U-deflation)
    X_res = X_res - t * P(:, a)';
    Y_res = Y_res - u * Q(:, a)';

    % 计算解释方差
    var_X = 1 - sum(X_res(:).^2) / sum(X0(:).^2);
    var_Y = 1 - sum(Y_res(:).^2) / sum(Y0(:).^2);
    fprintf('  LV %d: Var(X)=%.2f%%, Var(Y)=%.2f%%\n', a, var_X*100, var_Y*100);
end

% 计算重构矩阵
R_x = W * inv(P' * W);  % Input projection matrix
R_u = C * inv(Q' * C);  % Output reconstruction matrix (pseudoinverse)

% 残差
R_y = Y0 - U * Q';  % Output reconstruction residual

Model.T = T;
Model.U = U;
Model.W = W;
Model.C = C;
Model.P = P;
Model.Q = Q;
Model.R_x = R_x;
Model.R_u = R_u;
Model.R_y = R_y;
Model.X_std = Identification_data.X_std;
Model.Y_std = Identification_data.Y_std;

% ★ 核心修复：补全缺失的字段 ★
Model.X_mean = Identification_data.X_mean;
Model.Y_mean = Identification_data.Y_mean;
if isfield(Identification_data, 'W_y')
    Model.W_y = Identification_data.W_y;
else
    Model.W_y = ones(n_out, 1);
end
% ---------------------------------

%% 3. ARX Model Identification (仅使用控制段数据)
fprintf('build_dypls: Identifying ARX models for each latent variable...\n');

% ✅ 核心修正：计算训练起始点
start_k = ctrl_start + n_arx;

if start_k > Ble
    error('build_dypls: ARX order n_arx=%d 太大！控制段不足训练（ctrl_start=%d, Ble=%d）\n推荐: n_arx <= %d', ...
        n_arx, ctrl_start, Ble, Ble - ctrl_start - 10);
end

Model.Theta = cell(A_lv, 1);

for a = 1:A_lv
    Phi = [];
    Y_arx = [];

    for b = 1:N_batches
        idx = (b-1)*Ble + (1:Ble);
        t_b = T(idx, a);
        u_b = U(idx, a);

        % ✅ 修正：只使用控制段内有完整历史的数据
        for k = start_k:Ble
            u_lag = u_b(k-1:-1:k-n_arx);  % 历史输出分数 [k-1, k-2, ..., k-n_arx]
            t_lag = t_b(k-1:-1:k-n_arx);  % 历史输入分数 [k-1, k-2, ..., k-n_arx]
            Phi   = [Phi;  u_lag' t_lag'];
            Y_arx = [Y_arx; u_b(k)];
        end
    end

    % 岭回归求解
    lambda_a = Params.arx_lambda;
    theta_a = (Phi' * Phi + lambda_a * eye(2*n_arx)) \ (Phi' * Y_arx);

    Model.Theta{a} = theta_a;

    % 统计信息
    n_samples = size(Phi, 1);
    samples_per_batch = Ble - start_k + 1;
    fprintf('  LV %d: %d total samples (%d/batch, k=[%d,%d])\n', ...
        a, n_samples, samples_per_batch, start_k, Ble);

    % 计算条件数（诊断多重共线性）
    cond_num = cond(Phi' * Phi);
    if cond_num > 1e10
        warning('build_dypls: LV %d 的 ARX 回归矩阵条件数过大 (%.2e)，可能存在数值问题', a, cond_num);
    end
end

%% 4. Build Augmented State-Space Matrices
fprintf('build_dypls: Building augmented state-space model...\n');

% 每个潜变量的状态维度
n_state_per_lv = 2 * n_arx - 1;  % [u(k), u(k-1), ..., u(k-n_arx+1), t(k-1), ..., t(k-n_arx+1)]
nx = A_lv * n_state_per_lv;

barA = zeros(nx, nx);
barB = zeros(nx, A_lv);
barC = zeros(A_lv, nx);

for a = 1:A_lv
    theta = Model.Theta{a};
    alpha = theta(1:n_arx);      % ARX 系数（输出历史）
    beta = theta(n_arx+1:end);   % ARX 系数（输入历史）

    % 状态索引
    idx = (a-1)*n_state_per_lv + (1:n_state_per_lv);

    A_a = zeros(n_state_per_lv, n_state_per_lv);

    % 第一行：u(k+1) = alpha' * [u(k), ..., u(k-n_arx+1)] + beta' * [t(k), ..., t(k-n_arx+1)]
    A_a(1, 1:n_arx) = alpha';
    A_a(1, n_arx+1:n_state_per_lv) = beta(2:end)';  % beta(1) 对应 t(k)，在 B 中处理

    % 第 2 到 n_arx 行：移位寄存器（u 的历史）
    for i = 2:n_arx
        A_a(i, i-1) = 1;
    end

    % 第 n_arx+1 到 n_state_per_lv 行：移位寄存器（t 的历史）
    for i = n_arx+1:n_state_per_lv-1
        A_a(i+1, i) = 1;
    end

    barA(idx, idx) = A_a;

    % 输入矩阵 B（t(k) 的影响）
    B_a = zeros(n_state_per_lv, 1);
    B_a(1) = beta(1);        % u(k+1) 受 t(k) 影响
    B_a(n_arx+1) = 1;        % t(k) 存入历史第一位

    barB(idx, a) = B_a;

    % 输出矩阵 C（提取 u(k)）
    C_a = zeros(1, n_state_per_lv);
    C_a(1) = 1;

    barC(a, idx) = C_a;
end

Model.barA = barA;
Model.barB = barB;
Model.barC = barC;
Model.nx = nx;
Model.A_lv = A_lv;
Model.n_arx = n_arx;
% 扰动矩阵：每个潜变量的 ARX innovation 进入对应子系统的 u(k+1)
barE = zeros(nx, A_lv);
for a = 1:A_lv
    idx = (a-1)*n_state_per_lv + 1;
    barE(idx, a) = 1;
end
Model.barE = barE;

fprintf('build_dypls: State-space model built (nx=%d, nu=%d, ny=%d)\n', ...
    nx, A_lv, A_lv);
end
