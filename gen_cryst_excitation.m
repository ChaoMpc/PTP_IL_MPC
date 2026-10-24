function [Identification_data, Params] = gen_cryst_excitation(Params)
n_in_opt = Params.n_in_opt;
N_batches = Params.N_train_batches; Ble = Params.Ble;
N = N_batches * Ble; n_out = Params.n_out;

if isfield(Params,'rng_seed') && ~isempty(Params.rng_seed), rng(Params.rng_seed); end

X_abs_all = zeros(N, n_in_opt); Y_abs_all = zeros(N, n_out);

for b = 1:N_batches
    x_plant = Params.x0_nom;

    X_prbs = zeros(Ble, Params.n_in);
    for i = 1:n_in_opt
        u_max_space = Params.u_phys_max(i) - Params.u_base(i, :)';
        u_min_space = Params.u_base(i, :) - Params.u_phys_min(i);
        safe_max = u_max_space * 0.5;
        safe_min = u_min_space * 0.5;

        X_prbs(:, i) = prbs_signal(Ble, Params.prbs_periods(i), ...
            -safe_min, safe_max, Params.rng_seed + b * 100 + i);
    end

    X_prbs(1:Params.ctrl_start, :) = 0;
    X_batch = Params.u_base' + X_prbs;

    for k = 1:Ble
        X_batch(k,:) = max(min(X_batch(k,:), Params.u_phys_max'), Params.u_phys_min');
    end

    Y_batch = zeros(Ble, n_out);
    for k = 1:Ble
        [y_meas, x_plant] = cryst_plant(x_plant, X_batch(k,:)', Params, true);
        Y_batch(k,:) = y_meas';
    end

    start_row = (b-1) * Ble + 1; end_row = b * Ble;
    X_abs_all(start_row:end_row, :) = X_batch(:, 1:n_in_opt);
    Y_abs_all(start_row:end_row, :) = Y_batch;
end

U_base_all = repmat(Params.u_base_rep, N_batches, 1); % 120x5
Y_base_all = repmat(Params.y_base_rep, N_batches, 1); % 120x6

X_dev = X_abs_all - U_base_all;
Y_dev = Y_abs_all - Y_base_all;

X_mean = mean(X_dev, 1)'; X_std = std(X_dev, 0, 1)';
Y_mean = mean(Y_dev, 1)'; Y_std = std(Y_dev, 0, 1)';

if isfield(Params, 'W_y') && ~isempty(Params.W_y)
    Y_std = Y_std .* Params.W_y;
    Identification_data.W_y = Params.W_y;
else
    Identification_data.W_y = ones(n_out, 1);
end

X_std(X_std < 1e-12) = 1; Y_std(Y_std < 1e-12) = 1;

X0 = bsxfun(@rdivide, bsxfun(@minus, X_dev, X_mean'), X_std');
Y0 = bsxfun(@rdivide, bsxfun(@minus, Y_dev, Y_mean'), Y_std');

Identification_data.X_abs=X_abs_all; Identification_data.Y_abs=Y_abs_all;
Identification_data.X=X_dev; Identification_data.Y=Y_dev;
Identification_data.X0=X0; Identification_data.Y0=Y0;
Identification_data.X_mean=X_mean; Identification_data.X_std=X_std;
Identification_data.Y_mean=Y_mean; Identification_data.Y_std=Y_std;

Identification_data.u_start = Params.u_base(1:n_in_opt, 1);
Identification_data.y_start = Params.y_base(:, 1);

Identification_data.N=N; Identification_data.N_batches=N_batches;
Identification_data.Ble=Ble; Identification_data.time=(0:N-1)';
end

function u = prbs_signal(N, period, amp_low_vec, amp_high_vec, seed)
    if exist('OCTAVE_VERSION', 'builtin')
        old_state = rand("state");
        rand("state", seed);
        n_switches = ceil(N / period);
        bits = randi([0, 1], n_switches, 1);
        amps = 0.2 + 0.8 * rand(n_switches, 1);
        rand("state", old_state);
    else
        old_rng = rng;
        rng(seed, 'twister');
        n_switches = ceil(N / period);
        bits = randi([0, 1], n_switches, 1);
        amps = 0.2 + 0.8 * rand(n_switches, 1);
        rng(old_rng);
    end

    u = zeros(N, 1);
    for i = 1:n_switches
        s_idx = (i-1)*period + 1;
        e_idx = min(i*period, N);
        if bits(i) == 0
            u(s_idx:e_idx) = amp_low_vec(s_idx:e_idx) * amps(i);
        else
            u(s_idx:e_idx) = amp_high_vec(s_idx:e_idx) * amps(i);
        end
    end
end
