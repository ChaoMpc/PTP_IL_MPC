function [Identification_data, Params] = gen_prbs_excitation(Params)
% =========================================================================
% gen_prbs_excitation -- Generate multi-channel PRBS excitation data for
% the Wood-Berry binary distillation column (INCREMENTAL model).
%
% SCOPE
% -----
% This function ONLY generates identification data. It does NOT perform any
% constraint or reference-point conversion:
%
%   * Params.h_v_phys, Params.h_y_phys, Params.H_v, Params.H_y and
%     Params.Ref_points are ALREADY prepared in init_params, expressed in
%     the PHYSICAL DEVIATION space:
%         v = u - u_ss,   y = y - y_ss
%
%   * The standardized quantities X0, Y0 produced here are used ONLY
%     internally by build_dypls and estimate_uncertainty.
%
% =========================================================================

    %% =====================================================================
    % 1. Unpack parameters
    % =====================================================================
    N     = Params.N;
    Ts    = Params.Ts;
    n_in  = Params.n_in;
    n_out = Params.n_out;

    u_ss = Params.u_ss(:);
    y_ss = Params.y_ss(:);

    if ~isfield(Params, 'prbs_amp') || isempty(Params.prbs_amp)
        error('gen_prbs_excitation: Params.prbs_amp is required.');
    end
    prbs_amp = Params.prbs_amp(:);
    if numel(prbs_amp) ~= n_in || any(prbs_amp <= 0)
        error('gen_prbs_excitation: Params.prbs_amp must be positive, length n_in.');
    end

    if isfield(Params, 'prbs_bits') && ~isempty(Params.prbs_bits)
        prbs_bits = Params.prbs_bits;
    else
        prbs_bits = 10;
    end
    if isfield(Params, 'prbs_clock') && ~isempty(Params.prbs_clock)
        prbs_clock = Params.prbs_clock;
    else
        prbs_clock = 1;
    end
    if isfield(Params, 'plot_excitation') && ~isempty(Params.plot_excitation)
        plot_excitation = Params.plot_excitation;
    else
        plot_excitation = false;
    end

    %% =====================================================================
    % 2. Random seed
    % =====================================================================
    if isfield(Params, 'rng_seed') && ~isempty(Params.rng_seed)
        if exist('rng', 'builtin')
            rng(Params.rng_seed);
        else
            rand('state',  Params.rng_seed);
            randn('state', Params.rng_seed);
        end
        base_seed = Params.rng_seed;
    else
        base_seed = 1;
    end

    %% =====================================================================
    % 3. Validation
    % =====================================================================
    if N <= 0 || floor(N) ~= N
        error('gen_prbs_excitation: Params.N must be a positive integer.');
    end
    if Ts <= 0
        error('gen_prbs_excitation: Params.Ts must be positive.');
    end
    if Params.noise_var < 0
        error('gen_prbs_excitation: Params.noise_var must be nonnegative.');
    end

    [sys_ny, sys_nu] = size(Params.sys_d);
    if sys_nu ~= n_in || sys_ny ~= n_out
        error(['gen_prbs_excitation: sys_d size (%d x %d) is inconsistent ' ...
               'with n_in=%d, n_out=%d.'], sys_ny, sys_nu, n_out, n_in);
    end

    if ~isfield(Params, 'sys_d_d') || isempty(Params.sys_d_d)
        error('gen_prbs_excitation: Params.sys_d_d is missing.');
    end
    [dist_ny, dist_nu] = size(Params.sys_d_d);
    if dist_ny ~= n_out || dist_nu ~= 1
        error('gen_prbs_excitation: sys_d_d must be %d x 1.', n_out);
    end

    %% =====================================================================
    % 4. PRBS excitation -- DEVIATION form, one independent PRBS per input
    % =====================================================================
    X_dev = zeros(N, n_in);
    for i = 1:n_in
        amp_i = prbs_amp(i);
        X_dev(:, i) = prbs_signal(N, prbs_bits, -amp_i, amp_i, ...
                                  prbs_clock, base_seed + i);
    end

    %% =====================================================================
    % 5. Disturbance input
    % =====================================================================
    d = sqrt(Params.noise_var) * randn(N, 1);

    %% =====================================================================
    % 6. Simulate the INCREMENTAL plant
    % =====================================================================
    time = (0:N - 1)' * Ts;

    y_dev_nom  = lsim(Params.sys_d,   X_dev, time);
    y_dev_dist = lsim(Params.sys_d_d, d,     time);
    y_dev      = y_dev_nom + y_dev_dist;

    %% =====================================================================
    % 7. Absolute physical trajectories (bookkeeping / display only)
    % =====================================================================
    X_abs = X_dev + ones(N, 1) * u_ss';
    Y_abs = y_dev + ones(N, 1) * y_ss';

    %% =====================================================================
    % 8. Standardization on the DEVIATION data
    % ---------------------------------------------------------------------
    % X0, Y0 are used ONLY internally by build_dypls and
    % estimate_uncertainty. They are not constraint-space quantities.
    % =====================================================================
    X_mean = mean(X_dev, 1)';
    X_std  = std(X_dev, 0, 1)';
    Y_mean = mean(y_dev, 1)';
    Y_std  = std(y_dev, 0, 1)';

    X_std(X_std < 1e-12) = 1;
    Y_std(Y_std < 1e-12) = 1;

    X0 = (X_dev - ones(N, 1) * X_mean') ./ (ones(N, 1) * X_std');
    Y0 = (y_dev - ones(N, 1) * Y_mean') ./ (ones(N, 1) * Y_std');


    %% =====================================================================
    % 9. Package output
    % =====================================================================
    Identification_data.X      = X_dev;
    Identification_data.Y      = y_dev;
    Identification_data.X_abs  = X_abs;
    Identification_data.Y_abs  = Y_abs;
    Identification_data.X0     = X0;
    Identification_data.Y0     = Y0;
    Identification_data.X_mean = X_mean;
    Identification_data.X_std  = X_std;
    Identification_data.Y_mean = Y_mean;
    Identification_data.Y_std  = Y_std;
    Identification_data.u_ss   = u_ss;
    Identification_data.y_ss   = y_ss;
    Identification_data.N      = N;
    Identification_data.Ts     = Ts;
    Identification_data.n_in   = n_in;
    Identification_data.n_out  = n_out;
    Identification_data.time   = time;

    %% =====================================================================
    % 10. Optional plotting
    % =====================================================================
    if plot_excitation
        figure(100); clf;

        subplot(2, 1, 1);
        plot(0:N - 1, X_abs, 'LineWidth', 1.0);
        xlabel('Sample k'); ylabel('Input (physical)');
        title(sprintf('PRBS excitation (bits=%d, clock=%d, amp = 1%% u_{ss})', ...
            prbs_bits, prbs_clock));
        legend(arrayfun(@(i) sprintf('u_%d', i), 1:n_in, ...
            'UniformOutput', false));
        grid on;

        subplot(2, 1, 2);
        plot(0:N - 1, Y_abs, 'LineWidth', 1.0);
        xlabel('Sample k'); ylabel('Output (physical)');
        title('System outputs (physical)');
        legend(arrayfun(@(i) sprintf('y_%d', i), 1:n_out, ...
            'UniformOutput', false));
        grid on;

        drawnow;
    end

    %% =====================================================================
    % 11. Summary
    % =====================================================================
    fprintf('gen_prbs_excitation: PRBS data generated (DEVIATION form).\n');
    fprintf('  Samples          : %d\n', N);
    fprintf('  Sampling period  : %.4g min\n', Ts);
    fprintf('  I/O dimensions   : %d / %d\n', n_in, n_out);
    fprintf('  PRBS bits/clock  : %d / %d\n', prbs_bits, prbs_clock);
    fprintf('  PRBS amp (phys)  : [%s]\n', num2str(prbs_amp', '%.4f '));
    fprintf('  Noise variance   : %.4g\n', Params.noise_var);
    fprintf('  u_ss             : [%s]\n', num2str(u_ss', '%.4f '));
    fprintf('  y_ss             : [%s]\n', num2str(y_ss', '%.4f '));
    fprintf('--- Deviation statistics ---\n');
    fprintf('  X_mean           : [%s]\n', num2str(X_mean', '%.4e '));
    fprintf('  X_std            : [%s]\n', num2str(X_std',  '%.4e '));
    fprintf('  Y_mean           : [%s]\n', num2str(Y_mean', '%.4e '));
    fprintf('  Y_std            : [%s]\n', num2str(Y_std',  '%.4e '));
    fprintf(['Note: physical-deviation constraints and Ref_points were ' ...
             'prepared in init_params.\n']);
end

% =========================================================================
% Local function: prbs_signal
% =========================================================================
function u = prbs_signal(N, n_bits, amp_low, amp_high, clock_period, seed)

    if N <= 0 || floor(N) ~= N
        error('prbs_signal: N must be a positive integer.');
    end
    if n_bits < 2 || n_bits > 30 || floor(n_bits) ~= n_bits
        error('prbs_signal: n_bits must be an integer in [2, 30].');
    end
    if amp_high < amp_low
        error('prbs_signal: amp_high must be >= amp_low.');
    end
    if nargin < 5 || isempty(clock_period)
        clock_period = 1;
    end
    if nargin < 6 || isempty(seed)
        seed = 1;
    end
    if clock_period < 1 || floor(clock_period) ~= clock_period
        error('prbs_signal: clock_period must be a positive integer.');
    end

    switch n_bits
        case  2,  taps = [2 1];
        case  3,  taps = [3 2];
        case  4,  taps = [4 3];
        case  5,  taps = [5 3];
        case  6,  taps = [6 5];
        case  7,  taps = [7 6];
        case  8,  taps = [8 6 5 4];
        case  9,  taps = [9 5];
        case 10,  taps = [10 7];
        case 11,  taps = [11 9];
        case 12,  taps = [12 11 10 4];
        case 13,  taps = [13 12 11 8];
        case 14,  taps = [14 13 12 2];
        case 15,  taps = [15 14];
        case 16,  taps = [16 15 13 4];
        case 17,  taps = [17 14];
        case 18,  taps = [18 11];
        case 19,  taps = [19 18 17 14];
        case 20,  taps = [20 17];
        case 21,  taps = [21 19];
        case 22,  taps = [22 21];
        case 23,  taps = [23 18];
        case 24,  taps = [24 23 22 17];
        case 25,  taps = [25 22];
        case 26,  taps = [26 25 24 20];
        case 27,  taps = [27 26 25 22];
        case 28,  taps = [28 25];
        case 29,  taps = [29 27];
        case 30,  taps = [30 29 28 7];
        otherwise
            error('prbs_signal: no primitive polynomial for n_bits=%d.', ...
                n_bits);
    end

    register = zeros(1, n_bits);
    s = max(1, mod(seed, 2^n_bits - 1));
    for k = 1:n_bits
        register(k) = bitand(bitshift(s, -(k-1)), 1);
    end
    if all(register == 0)
        register(1) = 1;
    end

    n_bits_needed = ceil(N / clock_period) + 1;
    bits = zeros(n_bits_needed, 1);

    for k = 1:n_bits_needed
        bits(k) = register(end);

        fb = 0;
        for t = taps
            fb = bitxor(fb, register(t));
        end

        register(2:end) = register(1:end-1);
        register(1)     = fb;
    end

    u_bits = kron(bits, ones(clock_period, 1));
    u_bits = u_bits(1:N);

    u = amp_low + (amp_high - amp_low) * u_bits;
end
