function [Y0_sim, nrmses, vafs] = free_run_validation(Model, Val_data, Params, start_idx, window_len)
% =========================================================================
% free_run_validation -- 模型自由仿真 (Free-run)
% 支持从起点开始，前 n_arx 步使用真实数据作为种子，后续完全由模型递推。
% =========================================================================

if nargin < 5, start_idx = []; window_len = []; end

A_lv  = Model.A_lv;
n_arx = Model.n_arx;
Q_pls = Model.Q;
R_x   = Model.R_x;

N_val = size(Val_data.X0, 1);
T_val = Val_data.X0 * R_x;
Q_plus = pinv(Q_pls');
U_obs  = Val_data.Y0 * Q_plus;

U_sim = zeros(N_val, A_lv);

if ~isempty(start_idx) && ~isempty(window_len)
    start_idx = max(1, start_idx); % 允许从 1 (0时刻) 开始
    end_idx = min(start_idx + window_len - 1, N_val);

    % 1. 前置历史
    if start_idx > 1
        U_sim(1:start_idx-1, :) = U_obs(1:start_idx-1, :);
    end

    % 2. 窗口内的初始 Seed (前 n_arx 步使用真实值)
    seed_end = min(start_idx + n_arx - 1, end_idx);
    U_sim(start_idx:seed_end, :) = U_obs(start_idx:seed_end, :);

    % 3. 完全由模型递推
    for a = 1:A_lv
        theta_a = Model.Theta{a}(:);
        alpha = theta_a(1:n_arx);
        beta  = theta_a(n_arx + 1 : 2 * n_arx);
        for k = (seed_end + 1):end_idx
            u_lag = U_sim(k - 1 : -1 : k - n_arx, a);
            t_lag = T_val(k - 1 : -1 : k - n_arx, a);
            U_sim(k, a) = alpha' * u_lag + beta' * t_lag;
        end
    end

    if end_idx < N_val
        U_sim(end_idx+1:end, :) = U_obs(end_idx+1:end, :);
    end

    % 计算窗口指标
    Y0_sim_win = U_sim(start_idx:end_idx, :) * Q_pls';
    Y0_true_win = Val_data.Y0(start_idx:end_idx, :);
    nrmses = zeros(1, size(Y0_sim_win, 2)); vafs = zeros(1, size(Y0_sim_win, 2));
    for i = 1:size(Y0_sim_win, 2)
        err = Y0_true_win(:, i) - Y0_sim_win(:, i);
        denom = norm(Y0_true_win(:, i) - mean(Y0_true_win(:, i)));
        nrmses(i) = norm(err) / max(denom, eps);
        vafs(i)   = 100 * (1 - var(err) / max(var(Y0_true_win(:, i)), eps));
    end
else
    % 全局仿真
    U_sim(1:n_arx, :) = U_obs(1:n_arx, :);
    for a = 1:A_lv
        theta_a = Model.Theta{a}(:);
        alpha = theta_a(1:n_arx);
        beta  = theta_a(n_arx + 1 : 2 * n_arx);
        for k = (n_arx + 1):N_val
            u_lag = U_sim(k - 1 : -1 : k - n_arx, a);
            t_lag = T_val(k - 1 : -1 : k - n_arx, a);
            U_sim(k, a) = alpha' * u_lag + beta' * t_lag;
        end
    end
    Y0_sim_win = U_sim * Q_pls';
    nrmses = zeros(1, size(Y0_sim_win, 2)); vafs = zeros(1, size(Y0_sim_win, 2));
    for i = 1:size(Y0_sim_win, 2)
        err = Val_data.Y0(:, i) - Y0_sim_win(:, i);
        denom = norm(Val_data.Y0(:, i) - mean(Val_data.Y0(:, i)));
        nrmses(i) = norm(err) / max(denom, eps);
        vafs(i)   = 100 * (1 - var(err) / max(var(Val_data.Y0(:, i)), eps));
    end
end

Y0_sim = U_sim * Q_pls';
end
