% =========================================================================
% run_tuning.m -- Offline hyperparameter search for the crystallization
% DyPLS-ARX model.
% =========================================================================
clear; clc; close all;

if exist('OCTAVE_VERSION','builtin'), pkg load control optim; end

fprintf('=== Crystallization Hyperparameter Tuning ===\n');

%% 1. Base parameters
Params = init_params();

% 确认基准轨迹已正确加载
if isfield(Params, 'u_base') && isfield(Params, 'y_base')
    fprintf('  [OK] 已成功加载基准轨迹 (u_base, y_base).\n');
else
    error('未找到基准轨迹！请先运行 run_verify_profiles.m 生成 reference_trajectories.mat');
end

% Reproducible seeds
if isfield(Params,'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng','builtin'), rng(Params.rng_seed);
    else, rand('seed',Params.rng_seed); randn('seed',Params.rng_seed); end
end

% Ensure track_idx exists
if ~isfield(Params,'track_idx') || isempty(Params.track_idx)
    Params.track_idx = 0;
    fprintf('[warn] Params.track_idx not set; falling back to mean NRMSE.\n');
end

%% 2. Data (identification + independent validation)
fprintf('生成辨识数据 (基于轨迹偏差)...\n');
[Ident_data, Params] = gen_cryst_excitation(Params);

% ★ 核心检查：确保 Ident_data 包含所有必须的字段
required_fields = {'X0', 'Y0', 'X_mean', 'X_std', 'Y_mean', 'Y_std', 'W_y', 'N_batches', 'Ble'};
for i = 1:length(required_fields)
    if ~isfield(Ident_data, required_fields{i})
        error('Ident_data 缺失字段: %s。请检查 gen_cryst_excitation.m!', required_fields{i});
    end
end

fprintf('生成验证数据 (基于轨迹偏差)...\n');
[Val_data,  Params] = gen_cryst_validation_data(Params);

for i = 1:length(required_fields)
    if ~isfield(Val_data, required_fields{i})
        error('Val_data 缺失字段: %s。请检查 gen_cryst_validation_data.m!', required_fields{i});
    end
end

%% 3. Search
Params = tune_hyperparameters(Ident_data, Val_data, Params);

%% 4. Print copy-paste summary
fprintf('\n=== Tuning Finished ===\n');
fprintf('Update init_params.m with the selected values:\n');
fprintf('  Params.A_lv  = %d;\n', Params.A_lv);
fprintf('  Params.n_arx = %d;\n', Params.n_arx);
