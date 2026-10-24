function M = compute_metrics(Data, Params, Model, UncSets)
% 汇总闭环指标：约束违反、覆盖率、求解时间 (R1-Major 5,6)

N_batches = Params.N_batches;
N = Params.Ble;
H_v = Params.H_v; H_y = Params.H_y;
h_v_phys = Params.h_v_phys; h_y_phys = Params.h_y_phys;

M.max_violation_v = 0; M.max_violation_y = 0;
M.n_violations_v  = 0; M.n_violations_y  = 0;

for b = 1:N_batches
    for k = 1:N-1
        v = Data.V_hist(:, k, b);
        viol_v = H_v*v - h_v_phys;
        M.n_violations_v  = M.n_violations_v + sum(viol_v > 1e-6);
        M.max_violation_v = max(M.max_violation_v, max(viol_v));
    end
    for k = 1:N
        y = Data.Y_hist(:, k, b);
        viol_y = H_y*y - h_y_phys;
        M.n_violations_y  = M.n_violations_y + sum(viol_y > 1e-6);
        M.max_violation_y = max(M.max_violation_y, max(viol_y));
    end
end

% 覆盖率
flags = Data.eta_in_flags(:); flags = flags(flags>=0);
if isempty(flags)
    M.coverage_eta = 1;
else
    M.coverage_eta = mean(flags);
end
M.n_eta_samples = numel(flags);

flags = Data.r_in_flags(:); flags = flags(flags>=0);
if isempty(flags)
    M.coverage_r = 1;
else
    M.coverage_r = mean(flags);
end
M.n_r_samples = numel(flags);

M.qp_times = Data.QP_times(:);

fprintf('compute_metrics: %d/%d input violations, %d/%d output violations\n', ...
    M.n_violations_v, N_batches*(N-1), M.n_violations_y, N_batches*(N-1));
end
