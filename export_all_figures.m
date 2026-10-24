function export_all_figures(out_dir)
% =========================================================================
% export_all_figures -- Export specific figures to EPS/PDF/PNG.
% =========================================================================
if nargin < 1, out_dir = 'figures'; end
if ~exist(out_dir,'dir'), mkdir(out_dir); end

figs = findall(0,'Type','figure');
n = 0;
for k = 1:numel(figs)
    fh = figs(k);
    if ~ishandle(fh), continue; end

    nm = get(fh,'Name');
    if isempty(nm)
        nm = sprintf('figure_%d', fh);
    end

    if ~isempty(strfind(nm, 'Diagnostics')) || ~isempty(strfind(nm, 'Excitation'))
        continue;
    end

    fname = regexprep(nm, '[^a-zA-Z0-9]', '_');

    try
        print(fh, fullfile(out_dir, [fname '.eps']), '-depsc2', '-painters');
        print(fh, fullfile(out_dir, [fname '.pdf']), '-dpdf', '-painters');
        print(fh, fullfile(out_dir, [fname '.png']), '-dpng', '-r300');
        fprintf('  [OK] %s\n', fname);
        n = n + 1;
    catch ME
        fprintf('  [FAIL] %s: %s\n', fname, ME.message);
    end
end
fprintf('export_all_figures: %d figures exported to "%s"/\n', n, out_dir);
end
