function Params = tune_hyperparameters(Ident_data, Val_data, Params)
% =========================================================================
% tune_hyperparameters -- 基于独立验证集 free-run NRMSE 严格选择超参数
% =========================================================================
    fprintf('\n=== Hyperparameter Tuning (Cross-Validation) ===\n');

    % 1. 容错：获取 A_lv 搜索空间
    if isfield(Params, 'A_lv_cv_range') && ~isempty(Params.A_lv_cv_range)
        A_search = Params.A_lv_cv_range;
    else
        A_search = 1:Params.n_out;
    end
    % 确保 A 不会超过输出维数导致 Q 秩亏
    A_search = A_search(A_search <= Params.n_out);
    if isempty(A_search), A_search = Params.n_out; end

    % 获取 n_arx 搜索空间
    if isfield(Params, 'n_arx_cv_range') && ~isempty(Params.n_arx_cv_range)
        n_arx_search = Params.n_arx_cv_range;
    else
        n_arx_search = [4, 6, 8, 10, 12];
    end

    % 2. 打印表头 (重新对齐了列宽)
    fprintf('  A_lv | n_arx |  Val NRMSE (y1, y2)  |  Val VAF (y1, y2) %%\n');
    fprintf('  --------------------------------------------------------\n');

    best_nrmse = inf;
    best_n_arx = Params.n_arx;
    best_A = Params.A_lv;

    % 保存原始警告状态
    orig_warn_state = warning;
    warning('off', 'all');

    % 3. 嵌套遍历
    for A_try = A_search
        for n_arx_try = n_arx_search
            Params_temp = Params;
            Params_temp.A_lv = A_try;
            Params_temp.n_arx = n_arx_try;
            Params_temp.plot_model = false;

            try
                % 使用 evalc 捕获 build_dypls 内部的 fprintf，防止打乱表格
                if exist('evalc', 'builtin')
                    evalc('Model_temp = build_dypls(Ident_data, Params_temp);');
                else
                    Model_temp = build_dypls(Ident_data, Params_temp);
                end

                % 自由仿真
                [~, nrmses, vafs] = free_run_validation(Model_temp, Val_data, Params_temp);

                mean_nrmse = mean(nrmses);

                % 打印当前组合的结果 (使用 %9.4f 对齐)
                fprintf('   %2d  |  %2d   |  %9.4f, %9.4f  |  %7.2f, %7.2f\n', ...
                    A_try, n_arx_try, nrmses(1), nrmses(2), vafs(1), vafs(2));

                % 记录最优
                if mean_nrmse < best_nrmse
                    best_nrmse = mean_nrmse;
                    best_n_arx = n_arx_try;
                    best_A = A_try;
                end
            catch ME
                fprintf('   %2d  |  %2d   |  [FAILED] %s\n', A_try, n_arx_try, ME.message);
            end
        end
    end

    % 恢复警告状态
    warning(orig_warn_state);

    fprintf('  --------------------------------------------------------\n');
    fprintf('  Selected best: A_lv = %d, n_arx = %d (Mean NRMSE = %.4f)\n', ...
        best_A, best_n_arx, best_nrmse);

    % 4. 写回最优参数
    Params.A_lv  = best_A;
    Params.n_arx = best_n_arx;
end
