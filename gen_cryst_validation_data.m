function [Val_data, Params] = gen_cryst_validation_data(Params)
% =========================================================================
% gen_cryst_validation_data -- Generate an independent validation dataset.
% =========================================================================

saved_seed      = Params.rng_seed;
saved_N_batches = Params.N_train_batches;

Params.rng_seed        = Params.rng_seed_val;
Params.N_train_batches = Params.N_val_batches;

% 直接调用修改后的 excitation 函数
[Val_data, ~] = gen_cryst_excitation(Params);

Params.rng_seed        = saved_seed;
Params.N_train_batches = saved_N_batches;

Val_data.Description = 'Independent validation set';

fprintf('gen_cryst_validation_data: %d batches.\n', Params.N_val_batches);

end
