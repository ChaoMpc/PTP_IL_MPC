function [u_base, y_base, info] = verify_reference_trajectory(Params, mode, make_plots)
% =========================================================================
% verify_reference_trajectory -- 生成经验轨迹并开环验证
% =========================================================================
if nargin < 2, mode = 'constant'; end
if nargin < 3, make_plots = true; end

N = Params.Ble; n_in = Params.n_in; n_out = Params.n_out;
u_ss = Params.u_base(:, 1); t = (0:N-1) / max(N-1, 1);

%% ---- 1. 生成 u_base(k) ----
u_base = zeros(n_in, N);
switch lower(mode)
    case 'constant'
        u_base = repmat(u_ss, 1, N);
    case 'cooling'
        u_base(1,:) = 305 + (295 - 305) * t;
        for i = 2:n_in, u_base(i,:) = u_ss(i); end
    case 'cooling_aggr'
        u_base(1,:) = 308 + (290 - 308) * t;
        u_base(2,:) = 60  + (40  - 60 ) * t;
        u_base(3,:) = 1.0 + (2.0 - 1.0) * t;
        u_base(4,:) = 1.0 + (0.5 - 1.0) * t;
        u_base(5,:) = 300 + 50 * t;
        u_base(6,:) = 0.2 * ones(1, N);
        u_base(6, 1:min(10,N)) = 0.3;
    otherwise
        error('未知模式: %s', mode);
end
u_base = max(min(u_base, Params.u_phys_max(:)), Params.u_phys_min(:));

%% ---- 2. 开环仿真 ----
x = Params.x0_nom(:);
y_base = zeros(n_out, N + 1);
[y_base(:,1), x] = cryst_plant(x, u_ss, Params, false);
for k = 1:N
    [y_base(:,k+1), x] = cryst_plant(x, u_base(:,k), Params, false);
end

%% ---- 3. 诊断 ----
info = struct();
info.mode       = mode;
info.u_min      = min(u_base, [], 2);
info.u_max      = max(u_base, [], 2);
info.u_std      = std(u_base, 0, 2);
info.y_start    = y_base(:,1);
info.y_end      = y_base(:,end);
info.y_min      = min(y_base, [], 2);
info.y_max      = max(y_base, [], 2);
info.y_std      = std(y_base, 0, 2);

info.u_viol_max = max(u_base - repmat(Params.u_phys_max(:),1,N),   [], 2);
info.u_viol_min = max(repmat(Params.u_phys_min(:),1,N)  - u_base, [], 2);
info.y_viol_max = max(y_base - repmat(Params.y_phys_max(:),1,N+1), [], 2);
info.y_viol_min = max(repmat(Params.y_phys_min(:),1,N+1) - y_base, [], 2);

trk = Params.track_idx;
info.y_trk_at_Kp = y_base(trk, Params.Kp + 1);
info.max_abs_d2  = max(abs(diff(y_base(trk,:), 2)));

%% ---- 4. 打印 ----
in_names  = {'Tj_sp','Fc','Fanti','Fsol','Nstir','Fseed'};
out_names = {'T1','T2','T3','C_ATR','C_NIR','C_rho'};

fprintf('\n=================================================================\n');
fprintf('  Reference trajectory verification (mode = %s)\n', mode);
fprintf('=================================================================\n');

fprintf('\n--- u_base(k) 范围 (k=1..%d) ---\n', N);
fprintf('  %-8s | %10s | %10s | %10s | %10s\n','chan','min','max','std','u_ss');
for i = 1:n_in
    fprintf('  %-8s | %10.4f | %10.4f | %10.3e | %10.4f\n', ...
            in_names{i}, info.u_min(i), info.u_max(i), info.u_std(i), u_ss(i));
end

fprintf('\n--- y_base 范围 (k=0..%d) ---\n', N);
fprintf('  %-8s | %10s | %10s | %10s | %10s | %10s\n', ...
        'chan','y(0)','y(N)','min','max','drift');
for j = 1:n_out
    drift = info.y_end(j) - info.y_start(j);
    fprintf('  %-8s | %10.4f | %10.4f | %10.4f | %10.4f | %+10.3e\n', ...
            out_names{j}, info.y_start(j), info.y_end(j), ...
            info.y_min(j), info.y_max(j), drift);
end

fprintf('\n--- 追踪通道 %s 在 Kp 时刻的 y_base ---\n', out_names{trk});
for j = 1:numel(Params.Kp)
    k = Params.Kp(j);
    fprintf('  k=%3d : y_base=%.6f\n', k, y_base(trk,k+1));
end
fprintf('\n--- 平滑性 max|Δ²y_trk| = %.3e ---\n', info.max_abs_d2);

%% ---- 5. 绘图 ----
if make_plots
    tk = 0:N; tu = 1:N;

    % Fig 1: 输入
    figure('Name',sprintf('Ref input (%s)',mode),'Color','w','Position',[80 60 1400 700]);
    for i = 1:n_in
        subplot(2,3,i); hold on; grid on;
        plot(tu, u_base(i,:), 'b-', 'LineWidth', 1.8);
        plot([1 N],[Params.u_phys_max(i) Params.u_phys_max(i)], 'k--','LineWidth',0.8);
        plot([1 N],[Params.u_phys_min(i) Params.u_phys_min(i)], 'k--','LineWidth',0.8);
        plot([1 N],[u_ss(i) u_ss(i)], 'g:','LineWidth',1.0);
        xlabel('k'); xlim([1 N]); title(in_names{i},'FontSize',9);
        if i == 1
            % 规避 Octave 的 best 警告
            legend('u_{base}','max','min','u_{ss}','Location','northeast','FontSize',7);
        end
    end
    try, sgtitle(sprintf('Reference input profile (mode=%s)',mode),'FontWeight','bold'); catch, end

    % Fig 2: 输出
    figure('Name',sprintf('Ref output (%s)',mode),'Color','w','Position',[100 80 1400 800]);
    for j = 1:n_out
        subplot(2,3,j); hold on; grid on;
        plot(tk, y_base(j,:), 'r-', 'LineWidth', 1.8);
        plot([0 N],[Params.y_phys_max(j) Params.y_phys_max(j)], 'k--','LineWidth',0.8);
        plot([0 N],[Params.y_phys_min(j) Params.y_phys_min(j)], 'k--','LineWidth',0.8);
        if j == trk
            plot(Params.Kp+1, y_base(trk,Params.Kp+1), 'mo','MarkerSize',8,'MarkerFaceColor','m');
        end
        xlabel('k'); xlim([0 N]); title(out_names{j},'FontSize',9);
        if j == 1
            legend('y_{base}','max','min','Location','northeast','FontSize',7);
        end
    end
    try, sgtitle(sprintf('Reference output profile (mode=%s)',mode),'FontWeight','bold'); catch, end

    % Fig 3: 追踪通道放大
    figure('Name',sprintf('Tracking channel (%s)',mode),'Color','w','Position',[120 100 950 500]); hold on; grid on;
    plot(tk, y_base(trk,:), 'r-', 'LineWidth', 2.2);
    plot(Params.Kp+1, y_base(trk,Params.Kp+1), 'mo','MarkerSize',10,'MarkerFaceColor','m');
    for j = 1:numel(Params.Kp)
        k = Params.Kp(j);
        text(k+1, y_base(trk,k+1), sprintf('  k=%d, y=%.5f',k,y_base(trk,k+1)),'FontSize',9,'VerticalAlignment','bottom');
    end
    xlabel('k (min)'); ylabel(out_names{trk});
    title(sprintf('Tracking channel %s (mode=%s)', out_names{trk}, mode));
end
end
