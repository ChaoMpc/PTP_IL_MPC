function [Phi_x, x0_contrib, Phi_y, y0_const, Phi_v, v0_const] = ...
    build_lifting_matrices(N, nx, n_out, n_in, A_lv, barA, barB, barC, ...
        S_y, Q, S_v, R_x_inv, X_mean, Y_mean, x0)
% 构造 N-1 步的 lifting 矩阵：
%   x(k) = Phi_x(k) t_vec + A^{k} x0,  k = 1..N-1
%   y(k) = Y_mean + S_y Q Cbar x(k)
%   v(k) = X_mean + S_v R_x_inv t(k)
    nx_ = nx; n_state_total = A_lv*(N-1);
    Phi_x   = zeros(nx_*(N-1), n_state_total);
    Gamma_x = zeros(nx_*(N-1), nx_);
    A_pow   = cell(N,1); A_pow{1} = eye(nx_);
    for i = 2:N, A_pow{i} = barA*A_pow{i-1}; end
    for k = 1:N-1
        r = (k-1)*nx_+1 : k*nx_;
        for j = 1:k
            c = (j-1)*A_lv+1 : j*A_lv;
            Phi_x(r,c) = A_pow{k-j+1}*barB;
        end
        Gamma_x(r,:) = A_pow{k+1};
    end
    x0_contrib = Gamma_x*x0;

    C_y     = S_y*Q*barC;
    Phi_y   = zeros(n_out*(N-1), n_state_total);
    y0_const = zeros(n_out*(N-1),1);
    for k = 1:N-1
        r  = (k-1)*n_out+1 : k*n_out;
        sr = (k-1)*nx_+1    : k*nx_;
        Phi_y(r,:)   = C_y*Phi_x(sr,:);
        y0_const(r)  = Y_mean + C_y*x0_contrib(sr);
    end

    M_v      = S_v*R_x_inv;
    Phi_v    = zeros(n_in*(N-1), n_state_total);
    v0_const = zeros(n_in*(N-1),1);
    for k = 1:N-1
        r = (k-1)*n_in+1 : k*n_in;
        c = (k-1)*A_lv+1 : k*A_lv;
        Phi_v(r,c) = M_v;
        v0_const(r) = X_mean;
    end
end
