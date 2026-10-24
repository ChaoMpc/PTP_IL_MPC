function [y_meas, x_next] = cryst_plant(x, v, Params, is_noisy)
% =========================================================================
% cryst_plant -- 结晶动力学 (极致加速版，纯标量微积分)
% =========================================================================
    if nargin < 4, is_noisy = true; end

    % 仅测量模式：v = [] 时不积分，只返回传感器读数
    if isempty(v)
        T = x(1); C = x(3);
        y_true = zeros(6, 1);
        y_true(1) = T;
        y_true(2) = T + 0.08;
        y_true(3) = T - 0.06;
        y_true(4) = max(0, C);
        y_true(5) = max(0, C + 2.0e-4 + 1.5e-4 * (T - 300));
        y_true(6) = max(0, C - 1.5e-4 - 1.0e-4 * (T - 300));
        if is_noisy
            noise_std = [0.03; 0.04; 0.05; 2e-4; 3e-4; 2.5e-4];
            noise = noise_std .* (2*rand(6, 1) - 1);
            y_meas = max(0, y_true + noise);
        else
            y_meas = y_true;
        end
        x_next = x;
        return;
    end


    Ts_int = Params.Ts_int;
    n_steps = round(Params.Ts / Ts_int);

    % ---- 1. 提取物理参数并预先计算常数项 ----
    rho_c = Params.rho_c; k_v = Params.k_v; b = Params.b; g = Params.g;
    kb = exp(Params.ln_kb); kg = exp(Params.ln_kg);
    m_w = Params.m_w; C_pw = Params.C_pw; DHc = Params.DHc;
    F_c0 = Params.F_c0; N0 = Params.N0; nu_F = Params.nu_F; nu_N = Params.nu_N;
    V_sol = Params.V_sol;
    C_feed = Params.C_feed; k_anti = Params.k_anti; k_seed = Params.k_seed;

    tau_inv = 1 / Params.tau_j;

    % ---- 2. 提取输入为标量并预计算 ----
    Tj_sp  = v(1); Fc   = v(2); F_anti = v(3);
    F_sol  = v(4); N_stir = v(5); F_seed = v(6);

    UA = Params.UA0 * (Fc / F_c0)^nu_F * (N_stir / N0)^nu_N;
    anti_ratio = F_anti / (F_anti + F_sol + 10);
    dC_feed_base = F_sol * 1e-3 / m_w;
    seed_rate = F_seed * k_seed;

    % ---- 3. 提取状态为纯标量 ----
    T   = x(1); Tj  = x(2); C   = x(3);
    mu0 = x(4); mu1 = x(5); mu2 = x(6); mu3 = x(7);

    % ---- 4. 纯标量欧拉积分微循环 ----
    for i = 1:n_steps
        C_star = 1.5846e-5 * T^2 - 9.0567e-3 * T + 1.3066;
        C_star = C_star * (1 - k_anti * anti_ratio);
        dC_sat = C - C_star;

        if dC_sat > 0
            G = kg * dC_sat^g;
            B_rate = kb * dC_sat^b;
            if B_rate > 1e6, B_rate = 1e6; end
        else
            G = 0; B_rate = 0;
        end

        dmu0 = B_rate + seed_rate;
        dmu1 = G * mu0;
        dmu2 = 2 * G * mu1;
        dmu3 = 3 * G * mu2;

        dC_cryst = -rho_c * k_v * V_sol * (3 * G * mu2) / m_w;
        dC_dt = dC_cryst + (C_feed - C) * dC_feed_base;

        dT_dt = (UA * (Tj - T) - DHc * m_w * dC_cryst) / (m_w * C_pw);
        dTj_dt = (Tj_sp - Tj) * tau_inv;

        T   = T   + dT_dt * Ts_int;
        Tj  = Tj  + dTj_dt * Ts_int;
        C   = C   + dC_dt * Ts_int;
        mu0 = mu0 + dmu0 * Ts_int;
        mu1 = mu1 + dmu1 * Ts_int;
        mu2 = mu2 + dmu2 * Ts_int;
        mu3 = mu3 + dmu3 * Ts_int;

        % 【核心修复】：对 C 和所有 mu 严格非负截断
        if C < 1e-5, C = 1e-5; end
        if mu0 < 0, mu0 = 0; end
        if mu1 < 0, mu1 = 0; end
        if mu2 < 0, mu2 = 0; end
        if mu3 < 0, mu3 = 0; end
    end

    x_next = [T; Tj; C; mu0; mu1; mu2; mu3];

    if any(isnan(x_next)) || any(isinf(x_next))
        error('cryst_plant: Numerical instability detected.');
    end

    % ---- 5. 传感器模型 ----
    y_true = zeros(6, 1);
    y_true(1) = T;
    y_true(2) = T + 0.08;
    y_true(3) = T - 0.06;
    % 【核心修复】：确保传感器输出的物理量绝对不为负
    y_true(4) = max(0, C);
    y_true(5) = max(0, C + 2.0e-4 + 1.5e-4 * (T - 300));
    y_true(6) = max(0, C - 1.5e-4 - 1.0e-4 * (T - 300));

    if is_noisy
        noise_std = [0.03; 0.04; 0.05; 2e-4; 3e-4; 2.5e-4];
        noise = noise_std .* (2*rand(6, 1) - 1);
        y_meas = max(0, y_true + noise); % 加上噪声后依然截断非负
    else
        y_meas = y_true;
    end
end
