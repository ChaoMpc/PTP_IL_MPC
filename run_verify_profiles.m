% run_verify_profiles.m -- 对比参考轨迹模式
clear; clc; close all;
if exist('OCTAVE_VERSION','builtin'), pkg load control; end

Params = init_params();

% 1. 直接调用三次，不用循环，不用 eval，绝对不报错
[u_c, y_c, info_c] = verify_reference_trajectory(Params, 'constant', true);
[u_k, y_k, info_k] = verify_reference_trajectory(Params, 'cooling', true);
[u_a, y_a, info_a] = verify_reference_trajectory(Params, 'cooling_aggr', true);

% 2. 保存下来，供后续集成使用
save('reference_trajectories.mat', 'u_c', 'y_c', 'info_c', 'u_k', 'y_k', 'info_k', 'u_a', 'y_a', 'info_a', 'Params');

fprintf('\n\n=================================================================\n');
fprintf('  All 3 modes verified. Saved to reference_trajectories.mat\n');
fprintf('=================================================================\n');
