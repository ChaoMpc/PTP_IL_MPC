function Params = init_params()
% =========================================================================
% init_params -- Initialize parameters for the DyPLS-IL-MPC pipeline.
%
% Plant: Wood-Berry binary distillation column (2x2)
%
% SINGLE CONVERSION POINT
% -----------------------
%   All constraint and reference-point conversions from absolute physical
%   values to PHYSICAL DEVIATIONS are performed HERE, and only here:
%
%       v = u - u_ss   [lb/min]
%       y = y - y_ss   [wt%]
%
%   After this function returns, every downstream module
%   (gen_prbs_excitation, build_dypls, estimate_uncertainty, compute_tube,
%    run_ilmpc) operates EXCLUSIVELY on physical deviations. No module
%   re-standardizes the constraints, and no module divides the bounds by
%   X_std / Y_std.
%
%   Only plot_results re-adds u_ss / y_ss at the very end to display
%   absolute physical values.
% =========================================================================

    %% =====================================================================
    % 0. Control package
    % =====================================================================
    if exist('OCTAVE_VERSION', 'builtin')
        pkg load control;
    end

    %% =====================================================================
    % 1. Continuous-time plant -- Wood-Berry column
    % =====================================================================
    [numP_1, denP_1] = pade2(1);
    [numP_3, denP_3] = pade2(3);
    [numP_7, denP_7] = pade2(7);

    G11 = tf( 12.8 * numP_1, conv([16.7 1], denP_1));
    G12 = tf(-18.9 * numP_3, conv([21.0 1], denP_3));
    G21 = tf(  6.6 * numP_7, conv([10.9 1], denP_7));
    G22 = tf(-19.4 * numP_3, conv([14.4 1], denP_3));
    G   = [G11, G12; G21, G22];

    fprintf('init_params: Wood-Berry G(s) (2nd-order Pade):\n');
    disp(G);

    %% =====================================================================
    % 2. Disturbance model
    % =====================================================================
    Gd1 = tf(0.5, [10 1]);
    Gd2 = tf(0.5, [10 1]);
    Gd  = [Gd1; Gd2];

    %% =====================================================================
    % 3. Discretization
    % =====================================================================
    Params.Ts      = 1;                          % sampling period [min]
    Params.sys_d   = c2d(G,  Params.Ts);
    Params.sys_d_d = c2d(Gd, Params.Ts);

    %% =====================================================================
    % 4. Simulation and batch lengths
    % =====================================================================
    Params.N             = 4000;
    Params.Ble           = 60;
    Params.N_batches     = 15;
    Params.verbose_ilmpc = 1;

    %% =====================================================================
    % 5. System dimensions
    % =====================================================================
    Params.n_in  = 2;
    Params.n_out = 2;

    %% =====================================================================
    % 6. DyPLS identification parameters
    % =====================================================================
    Params.A_lv  = 2;
    Params.n_arx = 4;

    %% =====================================================================
    % 7. Steady-state operating point (Wood-Berry, 1973)
    % ---------------------------------------------------------------------
    %   u_ss = [R_ss ; S_ss]      lb/min
    %   y_ss = [xD_ss; xB_ss]     wt% methanol
    % =====================================================================
    Params.u_ss = [1.95; 1.71];
    Params.y_ss = [96.25; 0.50];

    %% =====================================================================
    % 8. Physical constraints -- ABSOLUTE -> PHYSICAL DEVIATION
    % ---------------------------------------------------------------------
    % The conversion is performed ONCE, here.
    % After this section:
    %     H_v * v <= h_v_phys,   H_y * y <= h_y_phys
    % with v = u - u_ss, y = y - y_ss, and
    %     h_v_phys = [ u_dev_max ; -u_dev_min ]
    %     h_y_phys = [ y_dev_max ; -y_dev_min ]
    %
    % No standardization (no ./X_std, ./Y_std) is applied.
    % =====================================================================
    % Absolute physical bounds
    Params.u_phys_min = [1.8525; 1.6245];   % 80%  of [R_ss ; S_ss]
    Params.u_phys_max = [2.0475; 1.7955];   % 120% of [R_ss ; S_ss]
    Params.y_phys_min = [95.0; 0.00];    % xD, xB lower bounds (wt%)
    Params.y_phys_max = [97.5; 1.00];    % xD, xB upper bounds (wt%)

    % Absolute physical -> physical deviation
    Params.u_dev_min = Params.u_phys_min - Params.u_ss;
    Params.u_dev_max = Params.u_phys_max - Params.u_ss;
    Params.y_dev_min = Params.y_phys_min - Params.y_ss;
    Params.y_dev_max = Params.y_phys_max - Params.y_ss;

    % Constraint matrices: H_v * v <= h_v_phys, H_y * y <= h_y_phys
    Params.H_v = [ eye(Params.n_in);  -eye(Params.n_in)  ];
    Params.H_y = [ eye(Params.n_out); -eye(Params.n_out) ];

    % Right-hand sides in physical deviation form
    Params.h_v_phys = [ Params.u_dev_max; -Params.u_dev_min ];
    Params.h_y_phys = [ Params.y_dev_max; -Params.y_dev_min ];

    % Legacy aliases: h_v / h_y now also denote physical deviations.
    Params.h_v = Params.h_v_phys;
    Params.h_y = Params.h_y_phys;

    %% =====================================================================
    % 9. Reference points -- ABSOLUTE -> PHYSICAL DEVIATION
    % ---------------------------------------------------------------------
    % Ref_points shares the same physical-deviation scale as y(k) inside
    % run_ilmpc:
    %     Ref_points = y_ref_phys - y_ss(1)   [wt% deviation]
    % =====================================================================
    Params.Kp         = [20, 40, 60];
    Params.y_ref_phys = [96.5, 96.7, 96.6];   % wt% methanol

    if numel(Params.Kp) ~= numel(Params.y_ref_phys)
        error('init_params: Kp and y_ref_phys must have the same length.');
    end
    if any(Params.Kp < 1) || any(Params.Kp > Params.Ble)
        error('init_params: Kp entries must lie in [1, Ble].');
    end
    for i = 1:numel(Params.y_ref_phys)
        if Params.y_ref_phys(i) < Params.y_phys_min(1) || ...
           Params.y_ref_phys(i) > Params.y_phys_max(1)
            error('init_params: y_ref_phys(%d)=%.3f outside [%.3f, %.3f].', ...
                i, Params.y_ref_phys(i), Params.y_phys_min(1), ...
                Params.y_phys_max(1));
        end
    end

    % Reference points in physical deviation
    Params.Ref_points = Params.y_ref_phys(:) - Params.y_ss(1);

    %% =====================================================================
    % 10. MPC weighting matrices
    % =====================================================================
    Params.Q_weight = 2.0 * eye(Params.n_out);
    Params.R_weight = 0.1 * eye(Params.A_lv);

    %% =====================================================================
    % 11. Prediction / control horizons
    % =====================================================================
    Params.P_horizon = Params.Ble;
    Params.M_horizon = Params.Ble;

    if Params.P_horizon > Params.Ble - 1
        warning('init_params: P_horizon > Ble-1; will be truncated.');
    end
    if Params.M_horizon > Params.P_horizon
        warning('init_params: M_horizon > P_horizon; will be truncated.');
    end

    %% =====================================================================
    % 12. Latent incremental bounds (physical-deviation-side auxiliary)
    % ---------------------------------------------------------------------
    % These bound the latent ILC increments, not the physical constraints.
    % =====================================================================
    Params.delta_t_max = 0.2;
    Params.ddt_max     = 0.2;

    %% =====================================================================
    % 13. PRBS excitation design
    % ---------------------------------------------------------------------
    % Amplitude = 1% of u_ss (per channel):
    %   R:  +/- 0.0195 lb/min
    %   S:  +/- 0.0171 lb/min
    % =====================================================================
    Params.prbs_amp   = 0.01 * Params.u_ss;   % [0.0195; 0.0171] lb/min
    Params.prbs_bits  = 10;                   % LFSR length (period 1023)
    Params.prbs_clock = 3;                    % samples per bit
    Params.noise_var  = 0.01;

    %% =====================================================================
    % 14. Uncertainty estimation
    % =====================================================================
    Params.safety_factor = 1.2;
    Params.q_level       = 0.95;

    %% =====================================================================
    % 15. Tube feedback
    % =====================================================================
    Params.tube_alpha = 0.2;
    Params.Q_lqr      = 10;
    Params.R_lqr      = 1;

    %% =====================================================================
    % 16. Plot switches
    % =====================================================================
    Params.plot_excitation = false;
    Params.plot_model      = false;
    Params.plot_results    = true;

    %% =====================================================================
    % 17. Random seed
    % =====================================================================
    Params.rng_seed = 26;

    %% =====================================================================
    % 18. Summary
    % =====================================================================
    fprintf('init_params: parameter initialization completed.\n');
    fprintf('  Steady state  u_ss  : [%s]  lb/min\n', ...
        num2str(Params.u_ss', '%.4f '));
    fprintf('  Steady state  y_ss  : [%s]  wt%%\n', ...
        num2str(Params.y_ss', '%.4f '));
    fprintf('  u_phys [min,max]    : [%s] / [%s]\n', ...
        num2str(Params.u_phys_min', '%.3f '), ...
        num2str(Params.u_phys_max', '%.3f '));
    fprintf('  y_phys [min,max]    : [%s] / [%s]\n', ...
        num2str(Params.y_phys_min', '%.3f '), ...
        num2str(Params.y_phys_max', '%.3f '));
    fprintf('  PRBS amp (physical) : [%s]\n', ...
        num2str(Params.prbs_amp', '%.4f '));
    fprintf('  PRBS bits / clock   : %d / %d\n', ...
        Params.prbs_bits, Params.prbs_clock);
    fprintf('  Batch length Ble    : %d\n', Params.Ble);
    fprintf('  LVs / ARX order     : %d / %d\n', ...
        Params.A_lv, Params.n_arx);
    fprintf('  Kp / y_ref_phys     : [%s] / [%s]\n', ...
        num2str(Params.Kp), num2str(Params.y_ref_phys, '%.3f '));
    fprintf('--- Physical-deviation constraints (single conversion point) ---\n');
    fprintf('  h_v_phys [upper; lower] : [%s] / [%s]\n', ...
        num2str(Params.u_dev_max', '%.4f '), ...
        num2str(Params.u_dev_min', '%.4f '));
    fprintf('  h_y_phys [upper; lower] : [%s] / [%s]\n', ...
        num2str(Params.y_dev_max', '%.4f '), ...
        num2str(Params.y_dev_min', '%.4f '));
    fprintf('  Ref_points (wt%% dev)   : [%s]\n', ...
        num2str(Params.Ref_points(:)', '%.4f '));
end

% =========================================================================
% Local function: pade2
% =========================================================================
function [num, den] = pade2(tau)
    num = [ tau^2, -6 * tau, 12 ];
    den = [ tau^2,  6 * tau, 12 ];
end
