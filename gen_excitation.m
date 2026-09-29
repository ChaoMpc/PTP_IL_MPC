function [Identification_data] = gen_excitation(Params)
    % ========================================================================
    % gen_excitation -- Generate excitation data for system identification.
    %
    % Tasks:
    %   (1) Generate n_in channels of random-step excitation u(k);
    %   (2) Simulate the nominal plant sys_d to obtain y_nom(k);
    %   (3) Superimpose disturbance via sys_d_d to obtain y_dist(k);
    %   (4) Column-wise standardize X and Y to obtain X0 and Y0;
    %   (5) Package the result into Identification_data for build_dypls.
    %
    % ------------------------------ Data convention -----------------------
    %   X(k, :) : physical input  vector at sample k (n_in)
    %   Y(k, :) : physical output vector at sample k (n_out)
    %   X0, Y0  : standardized data
    %   Time index k = 1, 2, ..., N
    %
    % ------------------------------ Inputs -------------------------------
    %   Params.N                  : number of samples
    %   Params.Ts                 : sampling period
    %   Params.n_in, n_out        : dimensions
    %   Params.sys_d              : nominal plant (n_out x n_in discrete LTI)
    %   Params.sys_d_d            : disturbance model (n_out x 1 discrete LTI)
    %   Params.exc_low, exc_high  : random-step amplitude range
    %   Params.min_pulse_width    : minimum step duration
    %   Params.max_pulse_width    : maximum step duration
    %   Params.noise_var          : disturbance variance (>= 0)
    %
    %   Optional:
    %     Params.plot_excitation  : logical, default false
    %     Params.rng_seed         : random seed, default unset
    %
    % ------------------------------ Outputs -------------------------------
    %   Identification_data.X, Y             : physical data
    %   Identification_data.X0, Y0           : standardized data
    %   Identification_data.X_mean, X_std    : input normalization
    %   Identification_data.Y_mean, Y_std    : output normalization
    %   Identification_data.N, Ts, n_in, n_out, time : metadata
    % ========================================================================

    %% ==================== 1. Read parameters ====================
    N = Params.N;
    Ts = Params.Ts;
    n_in = Params.n_in;
    n_out = Params.n_out;

    exc_low = Params.exc_low;
    exc_high = Params.exc_high;
    min_wid = Params.min_pulse_width;
    max_wid = Params.max_pulse_width;

    if isfield(Params, 'plot_excitation') &&~isempty(Params.plot_excitation)
        plot_excitation = Params.plot_excitation;
    else
        plot_excitation = false;
    end

    if isfield(Params, 'rng_seed') &&~isempty(Params.rng_seed)

        if exist('rng', 'builtin')
            rng(Params.rng_seed);
        else
            rand('seed', Params.rng_seed);
            randn('seed', Params.rng_seed);
        end

    end

    %% ==================== 2. Parameter validation ====================
    if N <= 0 || floor(N) ~= N
        error('gen_excitation: Params.N must be a positive integer.');
    end

    if Ts <= 0
        error('gen_excitation: Params.Ts must be positive.');
    end

    if min_wid <= 0 || max_wid < min_wid
        error('gen_excitation: invalid pulse widths (require 0 < min <= max).');
    end

    if floor(min_wid) ~= min_wid || floor(max_wid) ~= max_wid
        error('gen_excitation: pulse widths must be integers.');
    end

    if exc_high < exc_low
        error('gen_excitation: exc_high must be >= exc_low.');
    end

    if Params.noise_var < 0
        error('gen_excitation: Params.noise_var must be nonnegative.');
    end

    [sys_ny, sys_nu] = size(Params.sys_d);

    if sys_nu ~= n_in || sys_ny ~= n_out
        error('gen_excitation: sys_d size (%d x %d) inconsistent with n_in=%d, n_out=%d.', ...
            sys_ny, sys_nu, n_out, n_in);
    end

    [dist_ny, dist_nu] = size(Params.sys_d_d);

    if dist_ny ~= n_out || dist_nu ~= 1
        error('gen_excitation: sys_d_d must have %d outputs and 1 input.', n_out);
    end

    %% ==================== 3. Random-step excitation ====================
    X = zeros(N, n_in);

    for i = 1:n_in
        X(:, i) = random_step(N, [exc_low, exc_high], min_wid, max_wid);
    end

    %% ==================== 4. Disturbance ====================
    d = sqrt(Params.noise_var) * randn(N, 1);

    %% ==================== 5. Plant simulation ====================
    time = (0:N - 1)' * Ts;

    [y_nom, ~] = lsim(Params.sys_d, X, time);
    [y_dist, ~] = lsim(Params.sys_d_d, d, time);

    Y = y_nom + y_dist;

    if size(y_nom, 1) ~= N || size(y_nom, 2) ~= n_out
        error('gen_excitation: nominal output dimension mismatch.');
    end

    if size(y_dist, 1) ~= N || size(y_dist, 2) ~= n_out
        error('gen_excitation: disturbance output dimension mismatch.');
    end

    %% ==================== 6. Standardization ====================
    X_mean = mean(X, 1)';
    X_std = std(X, 0, 1)';

    Y_mean = mean(Y, 1)';
    Y_std = std(Y, 0, 1)';

    X_std(X_std < 1e-12) = 1;
    Y_std(Y_std < 1e-12) = 1;

    X0 = (X - ones(N, 1) * X_mean') ./ (ones(N, 1) * X_std');
    Y0 = (Y - ones(N, 1) * Y_mean') ./ (ones(N, 1) * Y_std');

    %% ==================== 7. Package data ====================
    Identification_data.X = X;
    Identification_data.Y = Y;
    Identification_data.X0 = X0;
    Identification_data.Y0 = Y0;

    Identification_data.X_mean = X_mean;
    Identification_data.X_std = X_std;
    Identification_data.Y_mean = Y_mean;
    Identification_data.Y_std = Y_std;

    Identification_data.N = N;
    Identification_data.Ts = Ts;
    Identification_data.n_in = n_in;
    Identification_data.n_out = n_out;
    Identification_data.time = time;

    %% ==================== 8. Optional plotting ====================
    if plot_excitation
        figure(100);
        clf;

        subplot(2, 1, 1);
        plot(0:N - 1, X, 'LineWidth', 1.2);
        xlabel('Time (k)');
        ylabel('Inputs');
        title('Random-step excitation signals');
        legend(arrayfun(@(i) sprintf('u_%d', i), 1:n_in, 'UniformOutput', false));
        grid on;

        subplot(2, 1, 2);
        plot(0:N - 1, Y, 'LineWidth', 1.2);
        xlabel('Time (k)');
        ylabel('Outputs');
        title('System outputs with disturbance');
        legend(arrayfun(@(i) sprintf('y_%d', i), 1:n_out, 'UniformOutput', false));
        grid on;

        drawnow;
    end

    %% ==================== 9. Summary ====================
    fprintf('gen_excitation: excitation data generated.\n');
    fprintf('  Number of samples : %d\n', N);
    fprintf('  Sampling period   : %.4g\n', Ts);
    fprintf('  Input/output dim  : %d / %d\n', n_in, n_out);
    fprintf('  Amplitude range   : [%.3f, %.3f]\n', exc_low, exc_high);
    fprintf('  Pulse width range : [%d, %d]\n', min_wid, max_wid);
    fprintf('  Noise variance    : %.4g\n', Params.noise_var);
end

% ========================================================================
% Local function: random_step
% ------------------------------------------------------------------------
% Generate a random-amplitude, random-duration step signal.
%
% Inputs:
%   N         : sequence length
%   amp_range : [low, high] uniform range for amplitude
%   min_width : minimum step duration
%   max_width : maximum step duration
%
% Output:
%   u         : N-by-1 step sequence
% ========================================================================
function u = random_step(N, amp_range, min_width, max_width)
    u = zeros(N, 1);
    i = 1;

    while i <= N
        amp = amp_range(1) + (amp_range(2) - amp_range(1)) * rand();
        width = randi([min_width, max_width]);
        last_index = min(i + width - 1, N);
        u(i:last_index) = amp;
        i = last_index + 1;
    end

end
