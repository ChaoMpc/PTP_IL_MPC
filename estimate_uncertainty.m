function UncSets = estimate_uncertainty(Model, Identification_data, Params)
% =========================================================================
% estimate_uncertainty -- 基于分位数的不确定性集合估计
%                         (仅统计控制段的残差)
% =========================================================================
% 输入：
%   Model: PLS-ARX 模型
%   Identification_data: 训练数据
%   Params: 参数结构体（包含 ctrl_start, q_level, safety_factor）
%
% 输出：
%   UncSets: 包含 eta_bound, r_bound 和椭球参数
% =========================================================================

A_lv  = Model.A_lv;
n_arx = Model.n_arx;
n_out = size(Model.Q, 1);

N         = size(Model.U, 1);
Ble       = Identification_data.Ble;
N_batches = N / Ble;

ctrl_start = Params.ctrl_start;

%% 1. ARX Innovation (eta) - 仅统计控制段
fprintf('estimate_uncertainty: Computing ARX innovations (control segment only)...\n');

% ✅ 核心修正：训练起始点与 build_dypls 保持一致
start_k = ctrl_start + n_arx;

if start_k > Ble
    error('estimate_uncertainty: n_arx 过大，无法统计残差');
end

samples_per_batch = Ble - start_k + 1;
total_samples = N_batches * samples_per_batch;

Eta = zeros(total_samples, A_lv);
sample_idx = 0;

for b = 1:N_batches
    idx   = (b-1)*Ble + (1:Ble);
    T_b   = Model.T(idx, :);
    U_b   = Model.U(idx, :);

    for a = 1:A_lv
        th    = Model.Theta{a};
        alpha = th(1:n_arx);
        beta  = th(n_arx+1:end);

        % ✅ 修正：只统计控制段内有完整历史的残差
        eta_batch = zeros(samples_per_batch, 1);

        for k = start_k:Ble
            u_lag = U_b(k-1:-1:k-n_arx, a);
            t_lag = T_b(k-1:-1:k-n_arx, a);

            u_pred = alpha' * u_lag + beta' * t_lag;
            eta_batch(k - start_k + 1) = U_b(k, a) - u_pred;
        end

        % 存储到全局矩阵
        row_start = (b-1)*samples_per_batch + 1;
        row_end   = b*samples_per_batch;
        Eta(row_start:row_end, a) = eta_batch;
    end
end

%% 2. Output Reconstruction Residual (r)
fprintf('estimate_uncertainty: Computing PLS reconstruction residuals...\n');

% ✅ 可选：也只统计控制段的重构残差
% 但由于 PLS 是静态映射，通常使用全部数据
% 这里我们保守地也只用控制段

R_all = Model.R_y;  % N x n_out

R = [];
for b = 1:N_batches
    idx = (b-1)*Ble + (1:Ble);
    % ✅ 只取控制段的残差
    R = [R; R_all(idx(ctrl_start:end), :)];  %#ok<AGROW>
end

%% 3. Compute Bounds via Quantile Method
safety = Params.safety_factor;  % 例如 1.2
qlev   = Params.q_level;        % 例如 0.95

eta_bound = safety * column_quantile(abs(Eta), qlev)';
r_bound   = safety * column_quantile(abs(R),   qlev)';

% 确保非零（避免除零）
eta_bound = max(eta_bound, 1e-10);
r_bound   = max(r_bound,   1e-10);

%% 4. Ellipsoidal Outer Approximation
UncSets.eta_bound = eta_bound;
UncSets.r_bound   = r_bound;

% 椭球参数（对角椭球）
UncSets.W_eta = A_lv * diag(eta_bound.^2);
UncSets.W_r   = n_out * diag(r_bound.^2);

%% 5. Coverage Statistics (on control segment data)
eta_in = all(abs(Eta) <= repmat(eta_bound', size(Eta,1), 1), 2);
r_in   = all(abs(R)   <= repmat(r_bound',   size(R,  1), 1), 2);

UncSets.eta_coverage = mean(eta_in);
UncSets.r_coverage   = mean(r_in);
UncSets.eta_samples  = Eta;
UncSets.r_samples    = R;

%% 6. Print Summary
fprintf('\n=== Uncertainty Set Summary ===\n');
fprintf('  Data range: k in [%d, %d] (control segment only)\n', start_k, Ble);
fprintf('  Total samples: %d (%d batches x %d samples/batch)\n', ...
    total_samples, N_batches, samples_per_batch);
fprintf('  Quantile level: %.2f%%, Safety factor: %.2f\n', qlev*100, safety);
fprintf('\n');
fprintf('  eta_bound (ARX innovation):\n');
for a = 1:A_lv
    fprintf('    LV %d: %.4e\n', a, eta_bound(a));
end
fprintf('\n');
fprintf('  r_bound (PLS residual):\n');
for j = 1:n_out
    fprintf('    Output %d: %.4e\n', j, r_bound(j));
end
fprintf('\n');
fprintf('  Coverage on training data:\n');
fprintf('    eta: %.2f%%\n', UncSets.eta_coverage * 100);
fprintf('    r  : %.2f%%\n', UncSets.r_coverage * 100);
fprintf('================================\n\n');

end

%% Local Function: Column-wise Quantile
function q = column_quantile(X, p)
% 计算每列的 p 分位数
    Xs = sort(X, 1);
    n  = size(Xs, 1);

    if n == 1
        q = Xs;
        return;
    end

    % 线性插值
    idx = p * (n-1) + 1;
    lo = floor(idx);
    hi = ceil(idx);
    w = idx - lo;

    if lo == hi
        q = Xs(lo,:);
    else
        q = (1-w)*Xs(lo,:) + w*Xs(hi,:);
    end
end
