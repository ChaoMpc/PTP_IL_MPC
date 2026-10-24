function report_pls_diagnostics(Model, Identification_data)
% =========================================================================
% report_pls_diagnostics -- Report PLS-level diagnostics.
% =========================================================================

X0 = Identification_data.X0;
Y0 = Identification_data.Y0;
T  = Model.T; U = Model.U;
Q  = Model.Q; A = Model.A_lv;

var_X_tot = sum(var(X0, 0, 1));
var_Y_tot = sum(var(Y0, 0, 1));

fprintf('\n--- PLS diagnostics ---\n');
for a = 1:A
    ev_X = 100 * var(T(:,a)) / max(var_X_tot, eps);
    ev_Y = 100 * var(U(:,a)) / max(var_Y_tot, eps);
    fprintf('  LV %d: EV_X=%.2f%%, EV_Y=%.2f%%\n', a, ev_X, ev_Y);
end
E_Y = Y0 - U * Q';
fprintf('  R_y: ||.||_F=%.4e, rank=%d\n', norm(E_Y,'fro'), rank(E_Y));
fprintf('  Cond(X0''X0/N)=%.4e\n', cond((X0'*X0)/size(X0,1)));
fprintf('  Cond(T''T/N)  =%.4e\n', cond((T'*T)/size(T,1)));

if isfield(Model,'P')
    err = Model.R_x' * Model.P - eye(A);
    fprintf('  ||R_x'' P - I||_F = %.4e (should be ~0)\n', norm(err,'fro'));
end

end
