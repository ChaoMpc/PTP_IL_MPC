function export_all_figures(Params, out_dir)
% =========================================================================
% export_all_figures -- Export all DyPLS-IL-MPC figures to lossless EPS.
%
% -------------------------------------------------------------------------
% Guarantees
% -------------------------------------------------------------------------
%   (1) Pure vector output:
%           -depsc2 + -painters  -> no rasterization of lines or text
%
%   (2) No positional shift:
%           PaperPosition is set in inches to match the on-screen figure
%           size exactly (screen_pixels / screen_dpi). Legends, axes and
%           labels therefore appear at the same relative location as on
%           screen.
%
%   (3) No clipping:
%           A small uniform margin is added to PaperSize so that long
%           xlabels and titles are never cut off.
%
%   (4) Robust across Octave versions:
%           -depsc2 is tried first; if the local Octave does not support it,
%           the code falls back to -depsc and then -deps.
%
% -------------------------------------------------------------------------
% Usage
% -------------------------------------------------------------------------
%   export_all_figures(Params)                % output to ./figures/
%   export_all_figures(Params, 'my_output')   % custom output directory
%
% -------------------------------------------------------------------------
% Figure identification
% -------------------------------------------------------------------------
%   Figures are found by their Name prefix (set in plot_results.m):
%       'Output Tracking'       -> fig1_tracking
%       'Control Inputs'        -> fig2_inputs
%       'ILC Convergence'       -> fig3_ilc
%       'DyPLS-ARX Fidelity'    -> fig4_fidelity
%       'Final Batch'           -> fig5_final_batch
%
%   If a figure does not exist (e.g. Model was not supplied), it is
%   silently skipped.
%
% -------------------------------------------------------------------------
% Files produced per figure
% -------------------------------------------------------------------------
%   <name>.eps   vector EPS, primary output for papers
%   <name>.pdf   vector PDF, convenience copy
%   <name>.png   300-dpi raster, for slides / quick preview
% =========================================================================

    %% =====================================================================
    % 0. Arguments
    % =====================================================================
    if nargin < 2 || isempty(out_dir)
        out_dir = 'figures';
    end

    %% =====================================================================
    % 1. Output directory
    % =====================================================================
    if ~exist(out_dir, 'dir')
        [ok, msg] = mkdir(out_dir);
        if ~ok
            error('export_all_figures: cannot create "%s": %s', out_dir, msg);
        end
    end

    %% =====================================================================
    % 2. Graphics toolkit -- MUST be Qt for reproducible rendering
    % =====================================================================
    is_octave = exist('OCTAVE_VERSION', 'builtin') ~= 0;

    if is_octave
        try
            graphics_toolkit('qt');
        catch
            warning(['export_all_figures: Qt toolkit unavailable; ' ...
                     'output may differ slightly from screen.']);
        end
    end

    %% =====================================================================
    % 3. Screen DPI -- used to convert pixels to inches
    % =====================================================================
    try
        screen_dpi = get(0, 'ScreenPixelsPerInch');
    catch
        screen_dpi = 96;
    end

    if isempty(screen_dpi) || ~isscalar(screen_dpi) || screen_dpi <= 0
        screen_dpi = 96;
    end

    fprintf('export_all_figures: screen DPI = %.1f\n', screen_dpi);

    %% =====================================================================
    % 4. Figure map: Name prefix -> output file base name
    % =====================================================================
    fig_map = {
        'Output Tracking',     'fig1_tracking';
        'Control Inputs',      'fig2_inputs';
        'ILC Convergence',     'fig3_ilc';
        'DyPLS-ARX Fidelity',  'fig4_fidelity';
        'Final Batch',         'fig5_final_batch';
    };

    %% =====================================================================
    % 5. Loop over figures
    % =====================================================================
    n_exported = 0;

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

        %% ---- 5.1 Pin the paper geometry to the on-screen size --------
        pos      = get(fh, 'Position');
        W_in     = pos(3) / screen_dpi;
        H_in     = pos(4) / screen_dpi;

        margin   = 0.05;      % inches, protects long labels from clipping

        set(fh, 'PaperUnits',         'inches');
        set(fh, 'PaperPositionMode',  'manual');
        set(fh, 'PaperPosition',      [margin, margin, W_in, H_in]);
        set(fh, 'PaperSize',          [W_in + 2*margin, H_in + 2*margin]);

        try
            set(fh, 'InvertHardcopy', 'off');
        catch
            % Some Octave versions lack this property; harmless to skip.
        end

        %% ---- 5.2 EPS: vector, primary output ------------------------
        eps_file = fullfile(out_dir, [file_base '.eps']);
        try
            print(fh, eps_file, '-depsc2', '-painters', '-r300');
            fprintf('  [OK]  %s\n', eps_file);
        catch
            try
                print(fh, eps_file, '-depsc', '-painters', '-r300');
                fprintf('  [OK]  %s  (fallback -depsc)\n', eps_file);
            catch
                try
                    print(fh, eps_file, '-deps', '-painters');
                    fprintf('  [OK]  %s  (fallback -deps)\n', eps_file);
                catch ME
                    warning(['export_all_figures: EPS export failed ' ...
                             'for "%s": %s'], name_prefix, ME.message);
                end
            end
        end

        %% ---- 5.3 PDF: vector, convenience copy ----------------------
        pdf_file = fullfile(out_dir, [file_base '.pdf']);
        try
            print(fh, pdf_file, '-dpdf', '-painters');
            fprintf('  [OK]  %s\n', pdf_file);
        catch ME
            warning(['export_all_figures: PDF export failed ' ...
                     'for "%s": %s'], name_prefix, ME.message);
        end

        %% ---- 5.4 PNG: high-resolution raster for slides -------------
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

    %% =====================================================================
    % 6. Summary
    % =====================================================================
    fprintf('\nexport_all_figures: %d figure(s) exported to "%s/".\n', ...
        n_exported, out_dir);
end

% =========================================================================
% Local function: find_figure_by_name
% -------------------------------------------------------------------------
% Return the handle of the first figure whose Name starts with the given
% prefix, or [] if none matches.
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
