function M = compute_metrics(Data, Params, Model, UncSets)
N_batches = Params.N_batches;
N         = Params.Ble;
n_in_opt  = Params.n_in_opt;

M.max_violation_v = 0; M.max_violation_y = 0;
M.n_violations_v  = 0; M.n_violations_y  = 0;
M.n_saturations   = 0;
M.coverage_eta    = NaN;
M.coverage_r      = NaN;
M.n_eta_samples   = 0;
M.n_r_samples     = 0;

total_v_samples = 0;
total_y_samples = 0;

for b = 1:N_batches
    for k = 1:N
        v = Data.V_hist(:,k,b);
        viol_v_up = v - Params.u_phys_max(1:n_in_opt);
        viol_v_lo = Params.u_phys_min(1:n_in_opt) - v;
        viol_v = max(viol_v_up, viol_v_lo);

        M.n_violations_v  = M.n_violations_v + sum(viol_v > 1e-6);
        M.max_violation_v = max(M.max_violation_v, max([0; viol_v]));

        sat_up = abs(v - Params.u_phys_max(1:n_in_opt)) < 1e-4;
        sat_lo = abs(v - Params.u_phys_min(1:n_in_opt)) < 1e-4;
        M.n_saturations = M.n_saturations + sum(sat_up | sat_lo);
    end
    total_v_samples = total_v_samples + N * n_in_opt;

    for k = 1:N
        y = Data.Y_hist(:,k,b);
        viol_y_up = y - Params.y_phys_max;
        viol_y_lo = Params.y_phys_min - y;
        viol_y = max(viol_y_up, viol_y_lo);

        M.n_violations_y  = M.n_violations_y + sum(viol_y > 1e-6);
        M.max_violation_y = max(M.max_violation_y, max([0; viol_y]));
    end
    total_y_samples = total_y_samples + N * Params.n_out;
end

if nargin >= 4 && ~isempty(UncSets) && isfield(Data,'eta_in_flags')
    ef = Data.eta_in_flags(:); ef = ef(ef ~= 0);
    rf = Data.r_in_flags(:);   rf = rf(rf ~= 0);

    if isempty(ef), M.coverage_eta = 1.0; else, M.coverage_eta = mean(ef); end
    if isempty(rf), M.coverage_r   = 1.0; else, M.coverage_r   = mean(rf); end

    M.n_eta_samples = numel(ef);
    M.n_r_samples   = numel(rf);
end

M.qp_times = Data.QP_times(:);

fprintf('compute_metrics:\n');
fprintf('  Input  Violations: %d / %d (Max: %.4e)\n', M.n_violations_v, total_v_samples, M.max_violation_v);
fprintf('  Output Violations: %d / %d (Max: %.4e)\n', M.n_violations_y, total_y_samples, M.max_violation_y);
fprintf('  Actuator Saturations: %d steps\n', M.n_saturations);

if isnan(M.coverage_eta)
    fprintf('  Coverage: not consider\n');
else
    fprintf('  Coverage: eta=%.2f%%, r=%.2f%%\n', 100*M.coverage_eta, 100*M.coverage_r);
end
end
