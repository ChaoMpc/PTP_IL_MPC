function [Val_data, Params] = gen_validation_data(Params)
% 独立验证数据生成（不同 PRBS 种子，不用于建模参数选择）
    saved_seed = Params.rng_seed;
    saved_N    = Params.N;
    Params.rng_seed = Params.rng_seed_val;
    Params.N        = Params.N_val;
    [Val_data, Params] = gen_prbs_excitation(Params);
    Params.rng_seed = saved_seed;
    Params.N        = saved_N;
end
