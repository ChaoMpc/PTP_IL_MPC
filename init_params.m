function Params = init_params()
    % ========================================================================
    % init_params -- Initialize all parameters for DyPLS-IL-MPC simulation.
    %
    % Covers:
    %   (1)  Continuous-time plant and disturbance model, discretized;
    %   (2)  Simulation and batch lengths;
    %   (3)  System dimensions;
    %   (4)  DyPLS identification parameters (A_lv, n_arx);
    %   (5)  Point-to-point reference points (mask);
    %   (6)  MPC weighting matrices;
    %   (7)  Prediction / control horizons;
    %   (8)  Input / output physical constraints;
    %   (9)  Excitation signal and noise;
    %   (10) Uncertainty estimation parameters;
    %   (11) Tube feedback pole placement;
    %   (12) Plot switches;
    %   (13) Random seed.
    %
    % Notes:
    %   This file only constructs the Params struct. All downstream functions
    %   (gen_excitation, build_dypls, estimate_uncertainty, compute_tube,
    %   run_ilmpc, plot_results) read fields from Params by name.
    % ========================================================================

    %% ==================== 1. Continuous-time plant ====================
    % Inputs: u1, u2; Outputs: y1, y2
    s = tf('s');

    G11 = 1.9 / (240 * s^2 + 31 * s + 1);
    G12 = 1.4 / (130 * s^2 + 23 * s + 1);
    G21 = 1.2 / (180 * s^2 + 28 * s + 1);
    G22 = 2.3 / (96 * s^2 + 20 * s + 1);

    G = [G11, G12;
        G21, G22];

    %% ==================== 2. Continuous-time disturbance ====================
    % A single disturbance input d acts on both outputs.
    Gd1 = 3 / (200 * s^2 + 30 * s + 1);
    Gd2 = 3 / (200 * s^2 + 30 * s + 1);

    Gd = [Gd1;
        Gd2];

    %% ==================== 3. Discretization ====================
    Params.Ts = 1;
    Params.sys_d = c2d(G, Params.Ts);
    Params.sys_d_d = c2d(Gd, Params.Ts);

    %% ==================== 4. Simulation and batch lengths ====================
    Params.N = 6000; % Number of samples for identification
    Params.Ble = 40; % Number of time steps per batch
    Params.N_batches = 15; % Total number of batches

    %% ==================== 5. System dimensions ====================
    Params.n_in = 2;
    Params.n_out = 2;

    %% ==================== 6. DyPLS identification parameters ====================
    Params.A_lv = 2; % Number of latent variables
    Params.n_arx = 4; % ARX order

    %% ==================== 7. Point-to-point reference points ====================
    % References are imposed on output y1 at specified time instants.
    Params.Kp = [10, 20, 30]; % Reference time indices (1..Ble)
    Params.Ref_points = [-0.1, 0.4, 0.1]; % Corresponding reference values

    % Consistency checks
    if numel(Params.Kp) ~= numel(Params.Ref_points)
        error('init_params: Kp and Ref_points must have the same length.');
    end

    if any(Params.Kp < 1) || any(Params.Kp > Params.Ble)
        error('init_params: entries of Kp must lie in [1, Ble].');
    end

    if any(floor(Params.Kp) ~= Params.Kp)
        error('init_params: entries of Kp must be integers.');
    end

    %% ==================== 8. MPC weighting matrices ====================
    Params.Q_weight = 2.0 * eye(Params.n_out); % Output tracking weight
    Params.R_weight = 0.5 * eye(Params.A_lv); % Latent-increment weight

    %% ==================== 9. Prediction / control horizons ====================
    Params.P_horizon = Params.Ble; % Prediction horizon
    Params.M_horizon = Params.Ble; % Control horizon

    if Params.P_horizon > Params.Ble - 1
        warning('init_params: P_horizon > Ble-1; run_ilmpc will truncate it.');
    end

    if Params.M_horizon > Params.P_horizon
        warning('init_params: M_horizon > P_horizon; run_ilmpc will truncate it.');
    end

    %% ==================== 10. Input / output constraints ====================
    % Constraints apply to physical input v and physical output y.
    Params.ub_v = 1 * ones(Params.n_in, 1);
    Params.lb_v = -1 * ones(Params.n_in, 1);
    Params.H_v = [eye(Params.n_in); -eye(Params.n_in)];
    Params.h_v = [Params.ub_v; -Params.lb_v];

    Params.ub_y = 1 * ones(Params.n_out, 1);
    Params.lb_y = -1 * ones(Params.n_out, 1);
    Params.H_y = [eye(Params.n_out); -eye(Params.n_out)];
    Params.h_y = [Params.ub_y; -Params.lb_y];

    % latent batch incremental input upper bound
    Params.delta_t_max = 0.01;
    % latent time incremental input upper bound
    Params.ddt_max = 0.01;

    %% ==================== 11. Excitation and noise ====================
    Params.exc_low = -1;
    Params.exc_high = 1;
    Params.min_pulse_width = 2;
    Params.max_pulse_width = 10;
    Params.noise_var = 0.1; % Disturbance variance during identification
    Params.noise_amptitude = 0.01;

    %% ==================== 12. Uncertainty estimation ====================
    Params.safety_factor = 1.2; % Conservativeness of uncertainty bounds
    Params.q_level = 0.95; % Quantile level

    %% ==================== 13. Tube feedback ====================
    Params.tube_alpha = 0.2; % Outer ellipsoid parameter (reserved)
    Dim = 2 * Params.n_arx - 1;
    Params.Q_lqr = 10;
    Params.R_lqr = 1;

    %% ==================== 14. Plot switches ====================
    Params.plot_excitation = false; % figure(100) inside gen_excitation
    Params.plot_model = false; % figure(99)  inside build_dypls
    Params.plot_results = true; % main.m calls plot_results

    %% ==================== 15. Random seed ====================
    Params.rng_seed = 26; % Set to [] to disable reproducibility

    %% ==================== 16. Plot switches ====================
    Params.plot_excitation = false;
    Params.plot_model      = false;
    Params.plot_results    = true;

    %% ==================== 17. Figure export ====================
    Params.export_figures  = true;
    Params.figures_dir     = 'figures';
    Params.fig_dpi         = 300;              % EPS/PNG 分辨率（>= 300）
    Params.fig_formats     = {'eps','pdf','png'};  % 或只保留需要的格式

    % 如果你确实需要纯矢量 EPS（可能较慢），打开下面这一行：
    % Params.fig_eps_painters = true;

    %% ==================== 18. Summary ====================
    fprintf('init_params: parameter initialization completed.\n');
    fprintf('  Batch length Ble       : %d\n', Params.Ble);
    fprintf('  Number of batches      : %d\n', Params.N_batches);
    fprintf('  Number of LVs A_lv     : %d\n', Params.A_lv);
    fprintf('  ARX order n_arx        : %d\n', Params.n_arx);
    fprintf('  Horizons P / M         : %d / %d\n', Params.P_horizon, Params.M_horizon);
    fprintf('  Reference time Kp      : [%s]\n', num2str(Params.Kp));
    fprintf('  Reference values       : [%s]\n', num2str(Params.Ref_points));
    fprintf('  safety_factor / q_level: %.2f / %.2f\n', ...
        Params.safety_factor, Params.q_level);
end
