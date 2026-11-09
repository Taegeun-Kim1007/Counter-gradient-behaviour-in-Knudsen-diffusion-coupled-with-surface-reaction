clc; clear; close all;

aa=50;
bb=30;
plot_x_length=1250;
plot_y_length=1110;
fontsize = 45;   % label + tick-number text size
linewidth = 6.0; % axis-box / grid line thickness

%% Read CSV
% The data file is read from the dataset folder data/ next to the code/ folder;
% set the environment variable COUNTER_GRADIENT_DATA to use another location.
data_dir = getenv('COUNTER_GRADIENT_DATA');
if isempty(data_dir); data_dir = fullfile('..', '..', 'data'); end
T = readmatrix(fullfile(data_dir, 'analytic', 'inversion_region_surface_grid200.csv'));

%% Extract data
aspect_ratio = T(:,2);
probability  = T(:,3);
Zdata        = T(:,7);

%% Remove only NaN for surface
valid = ~(isnan(aspect_ratio) | isnan(probability) | isnan(Zdata));
aspect_ratio = aspect_ratio(valid);
probability  = probability(valid);
Zdata        = Zdata(valid);

%% Triangulation for surface
tri = delaunay(aspect_ratio, probability);

tol = 1e-12;

%% ------------------------------
%% Max line for each AR
%% Exclude line points with AR=0 or Prob=0 or 1
%% ------------------------------
AR_unique_all = unique(aspect_ratio);
AR_unique = AR_unique_all(abs(AR_unique_all) > tol);

prob_max_line_AR = [];
Z_max_line_AR    = [];
AR_line_final    = [];

for i = 1:length(AR_unique)
    idx = (abs(aspect_ratio - AR_unique(i)) < tol) & ...
          (abs(probability) > tol) & ...
          (abs(probability - 1) > tol);

    if any(idx)
        Z_now = Zdata(idx);
        P_now = probability(idx);

        [Zmax, imax] = max(Z_now);

        AR_line_final(end+1,1)    = AR_unique(i);
        prob_max_line_AR(end+1,1) = P_now(imax);
        Z_max_line_AR(end+1,1)    = Zmax;
    end
end

%% ------------------------------
%% Max line for each Probability
%% Exclude line points with Prob=0 or 1, and AR=0
%% ------------------------------
P_unique_all = unique(probability);
P_unique = P_unique_all( abs(P_unique_all) > tol & abs(P_unique_all - 1) > tol );

AR_max_line_P = [];
Z_max_line_P  = [];
P_line_final  = [];

for i = 1:length(P_unique)
    idx = (abs(probability - P_unique(i)) < tol) & ...
          (abs(aspect_ratio) > tol);

    if any(idx)
        Z_now  = Zdata(idx);
        AR_now = aspect_ratio(idx);

        [Zmax, imax] = max(Z_now);

        P_line_final(end+1,1)  = P_unique(i);
        AR_max_line_P(end+1,1) = AR_now(imax);
        Z_max_line_P(end+1,1)  = Zmax;
    end
end

%% Output folder
save_dir = 'figures';
if ~exist(save_dir, 'dir'); mkdir(save_dir); end

% variables italic ({\it..}); text and the "m" subscript roman ({\rm..}). Notation from the paper.
% NOTE: MATLAB's tex interpreter needs the \rm to wrap the whole subscript group
% ({\rm_{m}}); writing {\rm m} works but inserts a visible space before the m,
% and {\rmm} is parsed as an unknown command and leaves the m italic.
label_sets = { ...
    struct('x','{\itL}{\rm_{m}}/(2{\ith})',           'y','{\it\kappa}',                       'dir',save_dir, 'name','figure5') ...
};

for k = 1:numel(label_sets)
    L = label_sets{k};

%% Plot
figure;
set(gcf, 'Position', [aa, bb, plot_x_length, plot_y_length]);
set(gca,'FontSize',fontsize,'FontName','Arial');   % Nature Physics: sans-serif (Arial)
set(gca,'LineWidth', linewidth)
hold on;

%% Surface: keep original contour/surface
trisurf(tri, aspect_ratio, probability, Zdata, ...
    'FaceColor','interp', ...
    'FaceAlpha',0.7, ...
    'EdgeColor','none');
shading interp;


% 'tex' interpreter so labels render in the axes font (Arial).
hx = xlabel(L.x,'Interpreter','tex','FontSize',fontsize);
hy = ylabel(L.y,'Interpreter','tex','FontSize',fontsize);
zlabel('Extent of the counter-gradient region (%)','Interpreter','tex','FontSize',fontsize)
colormap(jet);


ax = gca;
yticks(ax, [0 0.5 1.0]);
yticklabels(ax, {'0.0','0.5','1.0'});
xtickangle(ax, 0);
ytickangle(ax, 0);
ztickangle(ax, 0);
ax.XAxis.TickLabelGapOffset = 0;   % keep tick numbers snug to their tick marks
ax.YAxis.TickLabelGapOffset = 0;
ax.ZAxis.TickLabelGapOffset = 0;
zlim([0 22])
zticks(0:5.5:22)
grid on;
box on;
view(60,10.0);

% Tilt the x/y axis labels so each runs PARALLEL to its axis on screen.
drawnow;                                    % make sure the view/limits are applied
[vaz, vel] = view(ax);
Tv   = viewmtx(vaz, vel);
axpx = getpixelposition(ax);                % [x y w h] px
lims = [xlim(ax); ylim(ax); zlim(ax)];
rng  = diff(lims,1,2)';                      % data ranges
pba  = ax.PlotBoxAspectRatio;                % box shape
nb   = @(p) ((p - lims(:,1)')./rng) .* pba;  % data point -> normalized box coords
pj   = @(p) Tv * [nb(p)'; 1];
o  = pj([lims(1,1) lims(2,1) lims(3,1)]); o  = o/o(4);
ex = pj([lims(1,2) lims(2,1) lims(3,1)]); ex = ex/ex(4);   % along +x (L_m/(2h))
ey = pj([lims(1,1) lims(2,2) lims(3,1)]); ey = ey/ey(4);   % along +y (kappa)
wrap = @(a) a - 180*round(a/180);            % keep text upright (-90..90 deg)
hx.Rotation = -18.5;
hy.Rotation = 8;
%% Manual label position adjustment
% Set to the Position values read from the Property Inspector

% Show the automatic positions
disp('X label position:')
disp(hx.Position)

disp('Y label position:')
disp(hy.Position)

% Shift slightly from the automatic positions
hx.Position = [43.47583957522738,-0.154794245410361,-2.607187576242126];   % move the x label down
hy.Position = [56.354882312307836,0.904345506972131,-4.476037605069877];   % move the y label to the left

% Adjust the alignment if needed
hx.HorizontalAlignment = 'center';
hx.VerticalAlignment   = 'middle';

hy.HorizontalAlignment = 'center';
hy.VerticalAlignment   = 'middle';

%% Colorbar: preserve original axes exactly
drawnow;

fig = gcf;
ax  = gca;

fig.Units = 'pixels';
ax.Units  = 'pixels';

% Store the axes size and position before adding the colorbar
ax_pos = ax.Position;

% Add room for the colorbar on the right of the figure only
extra_width = 170;
fig.Position(3) = fig.Position(3) + extra_width;

% Create the colorbar
cb = colorbar(ax, 'eastoutside');

% Restore the axes that creating the colorbar has changed
ax.Position = ax_pos;

%% Colorbar position and size
cb.Units = 'pixels';

cb_width  = 28;
cb_height = 0.55 * ax_pos(4);
cb_gap    = 25;

cb.Position = [ ...
    ax_pos(1) + ax_pos(3) + cb_gap, ...
    ax_pos(2) + (ax_pos(4) - cb_height)/2, ...
    cb_width, ...
    cb_height];

%% Colorbar formatting
cb.FontSize = fontsize;
cb.FontName = 'Arial';
cb.LineWidth = linewidth;

cb.Label.String = 'Extent of the counter-gradient region (%)';
cb.Label.Interpreter = 'tex';
cb.Label.FontSize = fontsize;
cb.Label.FontName = 'Arial';

drawnow;

%% Save figure
% White background
set(gcf, 'Color', 'w');

% Save (.fig and high-resolution PNG)
savefig(gcf, fullfile(L.dir, [L.name, '.fig']));
exportgraphics(gcf, fullfile(L.dir, [L.name, '.png']), 'Resolution', 300);

end   % label_sets loop
