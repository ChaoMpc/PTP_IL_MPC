function Params = tune_hyperparameters(Ident_data, Val_data, Params)
% =========================================================================
% tune_hyperparameters -- Grid search over (A_lv, n_arx) using independent
% validation data.
% =========================================================================

fprintf('\n=== Hyperparameter Tuning (Cross-Validation on Control Phase) ===\n');

n_out     = Params.n_out;
n_in      = Params.n_in;
track_idx = Params.track_idx;
if track_idx > n_out, track_idx = 0; end

% ---- Search ranges ----
if isfield(Params,'A_lv_cv_range') && ~isempty(Params.A_lv_cv_range)
    A_search = Params.A_lv_cv_range;
else
    A_search = 1:min(n_out, 4);
end
A_search = A_search(A_search <= n_out);
A_search = A_search(A_search >= 1);
if isempty(A_search), A_search = 1:min(n_out, 4); end

if isfield(Params,'n_arx_cv_range') && ~isempty(Params.n_arx_cv_range)
    n_arx_search = Params.n_arx_cv_range;
else
    n_arx_search = [4, 6, 8, 10, 12];
end
n_arx_search = n_arx_search(n_arx_search < Params.Ble - 2);
n_arx_search = n_arx_search(n_arx_search >= 1);
if isempty(n_arx_search), n_arx_search = 4; end

% ---- Optional switches ----
compact      = get_opt(Params, 'tune_compact_mode',  false);
verbose_build= get_opt(Params, 'tune_verbose_build', false);
make_plots   = get_opt(Params, 'tune_make_plots',    true);
w_track      = get_opt(Params, 'tune_track_weight',  1.0);

if track_idx == 0, w_track = 0; end

% ---- Storage ----
nA = numel(A_search);
nN = numel(n_arx_search);
NRMSE_mat = NaN(nA, nN, n_out);
VAF_mat   = NaN(nA, nN, n_out);
Score_mat = NaN(nA, nN);
Time_mat  = NaN(nA, nN);

% ---- Table header ----
fprintf('\n  Legend: NRMSE = normalized RMS error; VAF = variance accounted for (%%)\n');
fprintf('  Track channel = %d (score weight = %.2f)\n\n', track_idx, w_track);

if compact
    fprintf('   A  | n_arx |  Track NRMSE | Track FR-VAF | Track 1-St VAF | Score | Time(s)\n');
    fprintf('  ----+-------+--------------+--------------+----------------+-------+--------\n');
else
    hdr1 = '   A  | n_arx |';
    hdr2 = '  ----+-------+';
    for j = 1:n_out
        hdr1 = [hdr1, sprintf('  y%d_NRMSE', j)];
        hdr2 = [hdr2, '-----------'];
    end
    for j = 1:n_out
        hdr1 = [hdr1, sprintf('  y%d_VAF%% ', j)];
        hdr2 = [hdr2, '-----------'];
    end
    hdr1 = [hdr1, '  MeanNRMSE | Mean1StepVAF |  Score  |  Time(s)'];
    hdr2 = [hdr2, '------------+--------------+---------+--------'];
    fprintf('%s\n%s\n', hdr1, hdr2);
end

% ---- Warnings off during loop ----
orig_warn = warning;
warning('off', 'all');

% ---- Main grid ----
best_score = inf;
best_A     = Params.A_lv;
best_n_arx = Params.n_arx;

% ★ 核心修复：验证窗口严格限制在控制阶段 ★
eval_start = Params.ctrl_start; % 从 60 分钟开始验证

for ia = 1:nA
    A_try = A_search(ia);
    for in = 1:nN
        n_arx_try = n_arx_search(in);
        if n_arx_try >= Params.Ble - 2, continue; end

        Params_t = Params;
        Params_t.A_lv  = A_try;
        Params_t.n_arx = n_arx_try;
        Params_t.plot_model = false;

        t_start = tic;
        try
            % Build DyPLS-ARX model
            if verbose_build
                Model_t = build_dypls(Ident_data, Params_t);
            else
                if exist('evalc','builtin')
                    % 在 Octave 下，用 evalc 屏蔽 build_dypls 的打印输出
                    evalc('Model_t = build_dypls(Ident_data, Params_t);');
                else
                    Model_t = build_dypls(Ident_data, Params_t);
                end
            end

            % 检查 Model_t 是否包含必需字段
            if ~isfield(Model_t, 'X_mean') || ~isfield(Model_t, 'W_y')
                error('build_dypls 构建的 Model 缺失核心字段(X_mean 或 W_y)');
            end

            % Per-batch free-run validation
            nrms_b   = zeros(1, n_out);
            vafs_b   = zeros(1, n_out);
            vafs1_b  = zeros(1, n_out);
            n_bat    = Params.N_val_batches;
            for b = 1:n_bat
                % 窗口起始为 60 分钟对应的行数
                s_b = (b-1) * Params.Ble + eval_start;
                l_b = Params.Ble - eval_start + 1;

                [Y0_sim_dummy, n_loc, v_loc, v1_loc] = free_run_validation( ...
                    Model_t, Val_data, Params_t, s_b, l_b);
                nrms_b  = nrms_b  + n_loc;
                vafs_b  = vafs_b  + v_loc;
                vafs1_b = vafs1_b + v1_loc;
            end
            nrmses = nrms_b / n_bat;
            vafs   = vafs_b / n_bat;
            vafs1  = vafs1_b / n_bat;

        catch ME
            Time_mat(ia, in) = toc(t_start);
            fprintf('  %2d  |  %2d   |  [FAILED]  %s\n', A_try, n_arx_try, ME.message);
            % 打印堆栈信息，方便定位
            for k = 1:length(ME.stack)
                fprintf('       > in %s (line %d)\n', ME.stack(k).name, ME.stack(k).line);
            end
            continue;
        end

        Time_mat(ia, in)   = toc(t_start);
        NRMSE_mat(ia,in,:) = nrmses;
        VAF_mat(ia,in,:)   = vafs;

        % Score: track-weighted + Complexity Penalty (AIC-like)
        complexity_penalty = 1e-4 * (A_try * 2 * n_arx_try);

        if track_idx > 0
            score = w_track * nrmses(track_idx) + (1 - w_track) * mean(nrmses) + complexity_penalty;
        else
            score = mean(nrmses) + complexity_penalty;
        end
        Score_mat(ia, in) = score;

        % Print row
        if compact
            track_n  = nrmses(max(track_idx,1));
            track_v  = vafs  (max(track_idx,1));
            track_v1 = vafs1 (max(track_idx,1));
            fprintf('  %2d  |  %2d   |  %12.4f |  %10.2f%% |  %12.2f%% |  %.4f |  %.3fs\n', ...
                A_try, n_arx_try, track_n, track_v, track_v1, score, Time_mat(ia,in));
        else
            line = sprintf('  %2d  |  %2d   |', A_try, n_arx_try);
            for j = 1:n_out
                line = [line, sprintf('  %9.4f', nrmses(j))]; %#ok<AGROW>
            end
            for j = 1:n_out
                line = [line, sprintf('  %9.2f', vafs(j))];   %#ok<AGROW>
            end
            line = [line, sprintf(' |  %8.4f |  %10.2f%% |  %6.4f |  %5.2fs', ...
                mean(nrmses), mean(vafs1), score, Time_mat(ia,in))];
            fprintf('%s\n', line);
        end

        % Best
        if score < best_score
            best_score = score;
            best_A     = A_try;
            best_n_arx = n_arx_try;
        end
    end
end

% Restore warnings
warning(orig_warn);

fprintf('\n  ----------------------------------------------------------------\n');
fprintf('  Best: A_lv = %d, n_arx = %d (score = %.6f)\n', ...
    best_A, best_n_arx, best_score);

% ---- Boundary warnings ----
if best_A == max(A_search)
    fprintf('  WARNING: best A_lv equals upper search bound (%d); consider\n', max(A_search));
    fprintf('           extending Params.A_lv_cv_range.\n');
end
if best_A == min(A_search) && min(A_search) == 1
    fprintf('  HINT: best A_lv = 1; increasing the search range may help.\n');
end
if best_n_arx == max(n_arx_search)
    fprintf('  WARNING: best n_arx equals upper search bound (%d); consider\n', max(n_arx_search));
    fprintf('           extending Params.n_arx_cv_range (must stay < Ble-2).\n');
end

% ---- Rebuild the best model and report its detailed per-channel table ----
fprintf('\n=== Detailed validation metrics for the selected model ===\n');
Params_best = Params;
Params_best.A_lv  = best_A;
Params_best.n_arx = best_n_arx;
if verbose_build
    Model_best = build_dypls(Ident_data, Params_best);
else
    if exist('evalc','builtin')
        evalc('Model_best = build_dypls(Ident_data, Params_best);');
    else
        Model_best = build_dypls(Ident_data, Params_best);
    end
end

nrms_b  = zeros(1, n_out);
vafs_b  = zeros(1, n_out);
vafs1_b = zeros(1, n_out);

for b = 1:Params.N_val_batches
    s_b = (b-1) * Params.Ble + eval_start;
    l_b = Params.Ble - eval_start + 1;

    [Y0_dummy, n_loc, v_loc, v1_loc] = free_run_validation( ...
        Model_best, Val_data, Params_best, s_b, l_b);
    nrms_b  = nrms_b  + n_loc;
    vafs_b  = vafs_b  + v_loc;
    vafs1_b = vafs1_b + v1_loc;
end
nrmses_best = nrms_b / Params.N_val_batches;
vafs_best   = vafs_b / Params.N_val_batches;
vafs1_best  = vafs1_b / Params.N_val_batches;

fprintf('  Channel |  FR-NRMSE| FR-VAF(%%) | 1-StepVAF(%%)|  Note\n');
fprintf('  --------+----------+-----------+--------------+----------------------------------\n');
for j = 1:n_out
    tag = '';
    if j == track_idx, tag = '  <-- tracked (Params.track_idx)'; end
    fprintf('  y%-6d |  %7.4f |  %7.2f  |  %10.2f  |%s\n', j, nrmses_best(j), vafs_best(j), vafs1_best(j), tag);
end
fprintf('  --------+----------+-----------+--------------+----------------------------------\n');
fprintf('  Mean    |  %7.4f |  %7.2f  |  %10.2f  |\n', mean(nrmses_best), mean(vafs_best), mean(vafs1_best));

% ---- Write best into the returned Params ----
Params.A_lv  = best_A;
Params.n_arx = best_n_arx;

% ---- Plots ----
if make_plots
    plot_tuning_grid(A_search, n_arx_search, NRMSE_mat, VAF_mat, ...
        Score_mat, track_idx, n_out);
end

end

%% =========================================================================
% Helper: get_opt
% =========================================================================
function v = get_opt(S, fname, default)
    if isfield(S, fname) && ~isempty(S.(fname))
        v = S.(fname);
    else
        v = default;
    end
end

%% =========================================================================
% Helper: plot_tuning_grid
%% =========================================================================
function plot_tuning_grid(A_search, n_arx_search, NRMSE_mat, VAF_mat, ...
        Score_mat, track_idx, n_out)

    nA = numel(A_search);
    nN = numel(n_arx_search);

    mean_nrmse = mean(NRMSE_mat, 3);
    mean_vaf   = mean(VAF_mat,   3);
    if track_idx > 0 && track_idx <= n_out
        track_nrmse = NRMSE_mat(:,:,track_idx);
        track_vaf   = VAF_mat(:,:,track_idx);
    else
        track_nrmse = mean_nrmse;
        track_vaf   = mean_vaf;
    end

    figure('Name','Hyperparameter Sensitivity','Color','w','Position',[100,100,1200,700]);

    subplot(2,2,1);
    imagesc(n_arx_search, A_search, track_nrmse); colorbar;
    xlabel('n_{arx}'); ylabel('A_{lv}');
    title(sprintf('Track NRMSE (channel %d)', track_idx));
    set(gca,'YDir','normal','XTick',n_arx_search,'YTick',A_search);
    for i = 1:nA
        for j = 1:nN
            text(n_arx_search(j), A_search(i), sprintf('%.3f', track_nrmse(i,j)), ...
                'HorizontalAlignment','center','FontSize',8);
        end
    end

    subplot(2,2,2);
    imagesc(n_arx_search, A_search, mean_nrmse); colorbar;
    xlabel('n_{arx}'); ylabel('A_{lv}');
    title('Mean NRMSE over all outputs');
    set(gca,'YDir','normal','XTick',n_arx_search,'YTick',A_search);
    for i = 1:nA
        for j = 1:nN
            text(n_arx_search(j), A_search(i), sprintf('%.3f', mean_nrmse(i,j)), ...
                'HorizontalAlignment','center','FontSize',8);
        end
    end

    subplot(2,2,3);
    imagesc(n_arx_search, A_search, track_vaf); colorbar;
    xlabel('n_{arx}'); ylabel('A_{lv}');
    title(sprintf('Track VAF %% (channel %d)', track_idx));
    set(gca,'YDir','normal','XTick',n_arx_search,'YTick',A_search);
    for i = 1:nA
        for j = 1:nN
            text(n_arx_search(j), A_search(i), sprintf('%.1f', track_vaf(i,j)), ...
                'HorizontalAlignment','center','FontSize',8);
        end
    end

    subplot(2,2,4);
    imagesc(n_arx_search, A_search, Score_mat); colorbar;
    xlabel('n_{arx}'); ylabel('A_{lv}');
    title('Selection score (lower = better)');
    set(gca,'YDir','normal','XTick',n_arx_search,'YTick',A_search);
    for i = 1:nA
        for j = 1:nN
            text(n_arx_search(j), A_search(i), sprintf('%.3f', Score_mat(i,j)), ...
                'HorizontalAlignment','center','FontSize',8);
        end
    end
end
