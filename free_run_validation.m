function [Y0_sim, nrmses, vafs, vafs_1step] = free_run_validation(Model, Val_data, Params, start_idx, window_len)
if nargin < 5, start_idx = []; window_len = []; end

if ~isfield(Model, 'X_mean') || ~isfield(Model, 'X_std') || ...
   ~isfield(Model, 'Y_mean') || ~isfield(Model, 'Y_std')
    Model.X_mean = Val_data.X_mean;
    Model.X_std  = Val_data.X_std;
    Model.Y_mean = Val_data.Y_mean;
    Model.Y_std  = Val_data.Y_std;
end
if ~isfield(Model, 'W_y')
    if isfield(Val_data, 'W_y'), Model.W_y = Val_data.W_y;
    else, Model.W_y = ones(Params.n_out, 1); end
end

A_lv  = Model.A_lv;
n_arx = Model.n_arx;
Q     = Model.Q;
N_val = size(Val_data.Y0, 1);
N_val_batches = N_val / Params.Ble;

X_base_val = repmat(Params.u_base(1:Params.n_in_opt, :)', N_val_batches, 1);
Y_base_val = repmat(Params.y_base(:, 2:end)', N_val_batches, 1);

X_val_dev = Val_data.X_abs - X_base_val;
Y_val_dev = Val_data.Y_abs - Y_base_val;

X_val_0   = bsxfun(@rdivide, bsxfun(@minus, X_val_dev, Model.X_mean'), Model.X_std');
Y_val_0   = bsxfun(@rdivide, bsxfun(@minus, Y_val_dev, Model.Y_mean'), Model.Y_std');

T_val = X_val_0 * Model.R_x;
U_obs = Y_val_0 * Model.R_u;

U_sim = zeros(N_val, A_lv);

% =========================================================================
% 1. Free-run (仅处理传入的窗口，并严防跨批次污染)
% =========================================================================
if ~isempty(start_idx) && ~isempty(window_len)
    start_idx = max(1, start_idx);

    % ★ 核心防越界保护：强制截断在当前批次内部
    batch_idx = floor((start_idx - 1) / Params.Ble); % 0-based 批次号
    batch_end = (batch_idx + 1) * Params.Ble;
    end_idx   = min(start_idx + window_len - 1, batch_end);

    seed_end  = min(start_idx + n_arx - 1, end_idx);
    U_sim(start_idx:seed_end, :) = U_obs(start_idx:seed_end, :);

    for a = 1:A_lv
        th = Model.Theta{a}; alpha = th(1:n_arx); beta = th(n_arx+1:end);
        for k = (seed_end+1):end_idx
            u_lag = U_sim(k-1:-1:k-n_arx, a);
            t_lag = T_val(k-1:-1:k-n_arx, a);
            U_sim(k,a) = alpha'*u_lag + beta'*t_lag;
        end
    end
    Y_sim_win  = U_sim(start_idx:end_idx,:) * Q';
    Y_true_win = Y_val_0(start_idx:end_idx,:);
else
    U_sim(1:n_arx,:) = U_obs(1:n_arx,:);
    for a = 1:A_lv
        th = Model.Theta{a}; alpha = th(1:n_arx); beta = th(n_arx+1:end);
        for k = (n_arx+1):N_val
            u_lag = U_sim(k-1:-1:k-n_arx, a);
            t_lag = T_val(k-1:-1:k-n_arx, a);
            U_sim(k,a) = alpha'*u_lag + beta'*t_lag;
        end
    end
    Y_sim_win  = U_sim * Q';
    Y_true_win = Y_val_0;
end

n_win = size(Y_true_win, 1);
W_y_mat_win = repmat(Model.W_y(:)', n_win, 1);
Y_true_dev = (Y_true_win .* Model.Y_std') ./ W_y_mat_win;
Y_sim_dev  = (Y_sim_win  .* Model.Y_std') ./ W_y_mat_win;

n_out = size(Y_sim_win, 2);
nrmses = zeros(1, n_out); vafs = zeros(1, n_out);

for i = 1:n_out
    err = Y_true_dev(:,i) - Y_sim_dev(:,i);
    denom = norm(Y_true_dev(:,i) - mean(Y_true_dev(:,i)));
    if denom < 1e-6, denom = max(abs(mean(Y_true_dev(:,i))), 1e-6); end
    nrmses(i) = norm(err) / denom;
    vafs(i)   = 100 * (1 - var(err)/max(var(Y_true_dev(:,i)), 1e-8));
end

% =========================================================================
% 2. One-step 指标
% =========================================================================
U_pred_1step = zeros(N_val, A_lv);
U_pred_1step(1:n_arx, :) = U_obs(1:n_arx, :);
for a = 1:A_lv
    theta = Model.Theta{a}; alpha = theta(1:n_arx); beta  = theta(n_arx+1:end);
    for k = (n_arx+1):N_val
        u_lag = U_obs(k-1:-1:k-n_arx, a);
        t_lag = T_val(k-1:-1:k-n_arx, a);
        U_pred_1step(k, a) = alpha' * u_lag + beta' * t_lag;
    end
end

Y_pred_1step = U_pred_1step * Q';
Y_true_1step = Y_val_0(n_arx+1:end, :);
Y_pred_1step = Y_pred_1step(n_arx+1:end, :);

n_1step = size(Y_true_1step, 1);
W_y_mat_1step = repmat(Model.W_y(:)', n_1step, 1);

Y_true_1step_dev = (Y_true_1step .* Model.Y_std') ./ W_y_mat_1step;
Y_pred_1step_dev = (Y_pred_1step .* Model.Y_std') ./ W_y_mat_1step;

vafs_1step = zeros(1, n_out);
for i = 1:n_out
    err_1st = Y_true_1step_dev(:, i) - Y_pred_1step_dev(:, i);
    vafs_1step(i) = 100 * (1 - var(err_1st) / max(var(Y_true_1step_dev(:, i)), 1e-8));
end

Y0_sim = U_sim * Q';
end
