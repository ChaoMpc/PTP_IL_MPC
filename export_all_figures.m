function export_all_figures(Params, out_dir)
% =========================================================================
% export_all_figures -- Export all DyPLS-IL-MPC figures to lossless EPS.
% =========================================================================

    %% 0. Arguments
    if nargin < 2 || isempty(out_dir)
        out_dir = 'figures';
    end

    %% 1. Output directory
    if ~exist(out_dir, 'dir')
        [ok, msg] = mkdir(out_dir);
        if ~ok
            error('export_all_figures: cannot create "%s": %s', out_dir, msg);
        end
    end

    %% 2. Graphics toolkit
    is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
    if is_octave
        try
            graphics_toolkit('qt');
        catch
            warning(['export_all_figures: Qt toolkit unavailable; ' ...
                     'output may differ slightly from screen.']);
        end
    end

    %% 3. Screen DPI
    try
        screen_dpi = get(0, 'ScreenPixelsPerInch');
    catch
        screen_dpi = 96;
    end
    if isempty(screen_dpi) || ~isscalar(screen_dpi) || screen_dpi <= 0
        screen_dpi = 96;
    end
    fprintf('export_all_figures: screen DPI = %.1f\n', screen_dpi);

    %% 4. Figure map
    % ★ 更新了 Figure 名称与文件名映射，反映了拆分与删除
    fig_map = {
        'Output Tracking y1',       'fig1a_tracking_y1';
        'Output Tracking y2',       'fig1b_tracking_y2';
        'Control Inputs v1',        'fig2a_inputs_v1';
        'Control Inputs v2',        'fig2b_inputs_v2';
        'ILC Convergence',          'fig3_ilc';
        'Free-run Validation',      'fig4_free_run';
        'Initial State Test y1',    'fig8a_init_y1';
        'Initial State Test y2',    'fig8b_init_y2'
    };

    %% 5. Loop over figures
    n_exported = 0;

    orig_warn_state = warning;
    warning('off', 'all');

    for k = 1:size(fig_map, 1)
        name_prefix = fig_map{k, 1};
        file_base   = fig_map{k, 2};

        fh = find_figure_by_name(name_prefix);
        if isempty(fh) || ~ishandle(fh)
            fprintf(['export_all_figures: no figure matching "%s" ' ...
                     '-- skipped.\n'], name_prefix);
            continue;
        end

        fprintf('\nexport_all_figures: exporting "%s" (handle %d)\n', ...
            name_prefix, fh);

        % ---- 5.1 Pin paper geometry to on-screen size -----------------
        pos  = get(fh, 'Position');
        W_in = pos(3) / screen_dpi;
        H_in = pos(4) / screen_dpi;

        margin = 0.05;

        set(fh, 'PaperUnits',        'inches');
        set(fh, 'PaperPositionMode', 'manual');
        set(fh, 'PaperPosition',     [margin, margin, W_in, H_in]);
        set(fh, 'PaperSize',         [W_in + 2*margin, H_in + 2*margin]);

        try
            set(fh, 'InvertHardcopy', 'off');
        catch
            % ignore
        end

        % ---- 5.2 EPS -------------------------
        eps_file = fullfile(out_dir, [file_base '.eps']);
        try
            print(fh, eps_file, '-depsc2', '-painters');
            fprintf('  [OK]  %s\n', eps_file);
        catch
            try
                print(fh, eps_file, '-depsc', '-painters');
                fprintf('  [OK]  %s  (fallback -depsc)\n', eps_file);
            catch
                try
                    print(fh, eps_file, '-deps');
                    fprintf('  [OK]  %s  (fallback -deps)\n', eps_file);
                catch ME
                    warning(['export_all_figures: EPS export failed ' ...
                             'for "%s": %s'], name_prefix, ME.message);
                end
            end
        end

        % ---- 5.3 PDF -----------------------
        pdf_file = fullfile(out_dir, [file_base '.pdf']);
        try
            print(fh, pdf_file, '-dpdf', '-painters');
            fprintf('  [OK]  %s\n', pdf_file);
        catch ME
            warning(['export_all_figures: PDF export failed ' ...
                     'for "%s": %s'], name_prefix, ME.message);
        end

        % ---- 5.4 PNG -------------------------
        png_file = fullfile(out_dir, [file_base '.png']);
        try
            print(fh, png_file, '-dpng', '-r300');
            fprintf('  [OK]  %s\n', png_file);
        catch ME
            warning(['export_all_figures: PNG export failed ' ...
                     'for "%s": %s'], name_prefix, ME.message);
        end

        n_exported = n_exported + 1;
    end

    warning(orig_warn_state);

    %% 6. Summary
    fprintf('\nexport_all_figures: %d figure(s) exported to "%s/".\n', ...
        n_exported, out_dir);
end

% =========================================================================
% Local function: find_figure_by_name
% =========================================================================
function fh = find_figure_by_name(name_prefix)
    fh = [];
    all_figs = findall(0, 'Type', 'figure');
    if isempty(all_figs)
        return;
    end
    for k = 1:numel(all_figs)
        if ~ishandle(all_figs(k))
            continue;
        end
        this_name = get(all_figs(k), 'Name');
        if ischar(this_name) && ...
                strncmp(this_name, name_prefix, numel(name_prefix))
            fh = all_figs(k);
            return;
        end
    end
end
