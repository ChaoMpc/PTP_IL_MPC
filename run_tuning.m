% =========================================================================
% run_tuning.m -- 独立运行的超参数寻优脚本
% 用于确定 PLS 潜变量数 (A_lv) 和 ARX 阶数 (n_arx)
% =========================================================================
clear; clc; close all;

if exist('OCTAVE_VERSION', 'builtin')
    pkg load control optim;
end

fprintf('=== Offline Hyperparameter Tuning ===\n');

% 1. 读取基础参数 (此时用的是 init_params 中的默认值)
Params = init_params();

% 随机种子设定
if isfield(Params, 'rng_seed') && ~isempty(Params.rng_seed)
    if exist('rng', 'builtin'), rng(Params.rng_seed);
    else, rand('seed', Params.rng_seed); randn('seed', Params.rng_seed); end
end

% 2. 生成数据
[Ident_data, Params] = gen_prbs_excitation(Params);
[Val_data, Params]   = gen_validation_data(Params);

% 3. 执行交叉验证遍历
Params = tune_hyperparameters(Ident_data, Val_data, Params);

fprintf('\n=== Tuning Finished ===\n');
fprintf('请将以下结果手动填入 init_params.m 中：\n');
fprintf('  Params.A_lv  = %d;\n', Params.A_lv);
fprintf('  Params.n_arx = %d;\n', Params.n_arx);
