% =========================================================================
% run_identification.m -- 离线辨识、Tube 计算与数据保存
% =========================================================================
clear; clc; close all;
if exist('OCTAVE_VERSION','builtin'), pkg load control optim; end

fprintf('=== 开始离线系统辨识 ===\n');
Params = init_params();
if isfield(Params,'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng','builtin'), rng(Params.rng_seed);
    else, rand('seed',Params.rng_seed); randn('seed',Params.rng_seed); end
end

% 1. 数据生成
fprintf('生成辨识数据...\n');
[Ident_data, Params] = gen_cryst_excitation(Params);
fprintf('生成验证数据...\n');
[Val_data, Params] = gen_cryst_validation_data(Params);

% 2. 模型构建
fprintf('构建 DyPLS-ARX 模型...\n');
Model = build_dypls(Ident_data, Params);
verify_pls_dimensionality(Ident_data, Model, Params);

% 3. Free-run 验证
eval_start = Params.ctrl_start + 1;
eval_len   = Params.Ble - Params.ctrl_start;
nrmses_sum = zeros(1, Params.n_out);
vafs_sum   = zeros(1, Params.n_out);
n_bat      = Params.N_val_batches;
Y0_sim = [];
for b = 1:n_bat
    s_b = (b-1) * Params.Ble + eval_start;
    [Y0_sim_dummy, n_loc, v_loc, ~] = free_run_validation(Model, Val_data, Params, s_b, eval_len);
    nrmses_sum = nrmses_sum + n_loc;
    vafs_sum   = vafs_sum + v_loc;
    if b == 1, Y0_sim = Y0_sim_dummy; end
end
CV_Data.Y0_sim     = Y0_sim;
CV_Data.nrmses     = nrmses_sum / n_bat;
CV_Data.vafs       = vafs_sum / n_bat;
CV_Data.start_idx  = eval_start;
CV_Data.window_len = eval_len;
fprintf('  [60~120 min Free-run] NRMSE = [%s]\n', num2str(CV_Data.nrmses,'%.4f '));
fprintf('  [60~120 min Free-run] VAF   = [%s] %%\n', num2str(CV_Data.vafs,'%.4f '));

% 4. 不确定性与 Tube 计算
fprintf('计算不确定性与 Tube 参数...\n');
UncSets = estimate_uncertainty(Model, Ident_data, Params);
Tube = compute_tube(Ident_data, Model, UncSets, Params);


% 5. 保存所有结果
fprintf('保存模型至 identified_model.mat ...\n');
save('identified_model.mat', 'Ident_data', 'Val_data', 'Model', ...
     'UncSets', 'Tube', 'CV_Data', 'Params');

fprintf('=== 辨识完成，数据已保存！后续可直接运行 main.m ===\n');
