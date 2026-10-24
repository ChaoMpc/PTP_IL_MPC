function report_pls_diagnostics(Model, Identification_data)
% =========================================================================
% report_pls_diagnostics -- 报告 PLS 层诊断量（审稿人 R1-Major 4）。
%
% 目的
% ----
%   原稿两个算例都是 m_v = m_y = A = 2，PLS 只是可逆旋转，掩盖了
%   PLS 真正的好处（降维 / 缓解共线性 / 改善数值条件）。修改稿中
%   Wood–Berry 也保持 m=A=2，所以本函数把 PLS 层应当报告的所有量都
%   打印出来，供审稿人确认：
%     - 每分量的解释方差 EV_X / EV_Y
%     - PLS 截断残差 R_y 范数
%     - 原始协方差矩阵条件数 vs 潜空间条件数
%     - 输入重构伪逆的秩 / 零空间维度（R1-Minor 17）
%
% 调用方式
% --------
%   推荐在 build_dypls.m 的 plot_model 分支中调用：
%       if plot_model, report_pls_diagnostics(Model, Identification_data); end
%   也可以由 main.m 在 build_dypls 之后手动调用一次。
% =========================================================================

if ~isfield(Model,'T') || ~isfield(Model,'U') || ~isfield(Model,'Q') ...
|| ~isfield(Model,'R_x') || ~isfield(Model,'A_lv')
    error('report_pls_diagnostics: Model missing required fields (T,U,Q,R_x,A_lv).');
end

X0 = Identification_data.X0;
Y0 = Identification_data.Y0;
T  = Model.T;
U  = Model.U;
Q  = Model.Q;
A  = Model.A_lv;

fprintf('\n--- PLS diagnostics ---\n');

% (1) 每分量解释方差
var_X_total = sum(var(X0, 0, 1));
var_Y_total = sum(var(Y0, 0, 1));
fprintf('Per-component variance explained:\n');
for a = 1:A
    ev_X = 100 * var(T(:, a)) / max(var_X_total, eps);
    ev_Y = 100 * var(U(:, a)) / max(var_Y_total, eps);
    fprintf('  LV %d: var(T_a)=%.4e, var(U_a)=%.4e, EV_X=%.2f%%, EV_Y=%.2f%%\n', ...
        a, var(T(:, a)), var(U(:, a)), ev_X, ev_Y);
end

% (2) PLS 截断残差
E_Y = Y0 - U * Q';
fprintf('PLS truncation residual (R_y): ||.||_F = %.4e, max|.| = %.4e, rank = %d\n', ...
    norm(E_Y, 'fro'), max(abs(E_Y(:))), rank(E_Y));

% (3) 条件数对比（原空间 vs 潜空间）
W_orig = (X0' * X0) / size(X0, 1);
fprintf('Cond(X0''X0 / N) = %.4e  (original space)\n', cond(W_orig));

TT = T' * T / size(T, 1);
fprintf('Cond(T''T / N)   = %.4e  (latent space)\n', cond(TT));

if size(Q, 2) == A
    fprintf('Cond(Q''Q)      = %.4e  (output latent)\n', cond(Q' * Q));
end

% (4) 输入重构伪逆 / 零空间维度（R1-Minor 17）
R_x = Model.R_x;
[nv, nA] = size(R_x);
fprintf('R_x: size = %d x %d, rank = %d\n', nv, nA, rank(R_x));
if nv > nA
    fprintf('Input null-space dim = %d  (A = %d < m_v = %d)\n', nv - nA, nA, nv);
else
    fprintf('Input null-space dim = 0  (A = m_v)\n');
end

% (5) 输入重构误差（随机检验）
%   v = X_mean + S_v R_x_inv t，随后 t_re = R_x' S_v^-1 (v - X_mean)
%   在 A = m_v 时误差为 0；在 A < m_v 时为 0（构造使然），但 v 只在潜子空间内
%   这里只是验证 R_x' P = I_A 的恒等式。
if isfield(Model, 'P')
    errP = R_x' * Model.P - eye(A);
    fprintf('||R_x'' P - I_A||_F = %.4e  (should be ~0)\n', norm(errP, 'fro'));
end

fprintf('--- end PLS diagnostics ---\n');
end
