function result = mercury_bowshock_batch(dataDir, outputDir)
%MERCURY_BOWSHOCK_BATCH Validate data, seed crossings, and export figures.
%   RESULT = MERCURY_BOWSHOCK_BATCH(DATA_DIR, OUTPUT_DIR) never modifies the
%   source MAT files. It crops each file to its filename's natural UTC day.
%   Periodic calibration-like bursts are retained in the source data but
%   masked in figures and excluded from automatic confidence scoring.

scriptDir = fileparts(mfilename('fullpath'));
mercuryDir = fileparts(scriptDir);
if nargin < 1 || isempty(dataDir)
    dataDir = fullfile(mercuryDir,'201312_01s');
end
if nargin < 2 || isempty(outputDir)
    outputDir = scriptDir;
end
dataDir = char(dataDir);
outputDir = char(outputDir);
assert(isfolder(dataDir),'MercuryBowShock:MissingData', ...
    'Data directory does not exist: %s',dataDir);
if ~isfolder(outputDir)
    mkdir(outputDir);
end

files = dir(fullfile(dataDir,'201312*_01s.mat'));
names = string({files.name});
validName = ~cellfun('isempty',regexp(cellstr(names), ...
    '^201312\d{2}_01s\.mat$','once'));
files = files(validName);
[~,order] = sort({files.name});
files = files(order);
assert(numel(files) == 31,'MercuryBowShock:FileCount', ...
    'Expected 31 daily MAT files, found %d.',numel(files));

metadata = table('Size',[31 9], ...
    'VariableTypes',{'string','double','datetime','double','datetime', ...
    'datetime','double','double','double'}, ...
    'VariableNames',{'File','Bytes','Modified','Rows','FirstUTC','LastUTC', ...
    'ExtremeCount','ExcludedCount','InternalGapCount'});
metadata.FirstUTC.TimeZone = 'UTC';
metadata.LastUTC.TimeZone = 'UTC';
crossings = emptyCrossingTable();
exclusions = emptyExclusionTable();
totalRows = 0;
previousEnd = -Inf;
boundaryOverlapCount = 0;
candidateExcludedCount = 0;

for k = 1:numel(files)
    filePath = fullfile(files(k).folder,files(k).name);
    info = whos('-file',filePath);
    hit = strcmp({info.name},'data_mso');
    assert(nnz(hit)==1 && strcmp(info(hit).class,'double') && ...
        numel(info(hit).size)==2 && info(hit).size(2)==7, ...
        'MercuryBowShock:Schema','%s must contain numeric data_mso with 7 columns.', ...
        files(k).name);
    loaded = load(filePath,'data_mso');
    raw = loaded.data_mso;
    assert(all(isfinite(raw),'all'),'MercuryBowShock:NonFinite', ...
        '%s contains non-finite values.',files(k).name);
    dt = diff(raw(:,1))*86400;
    assert(all(dt>0),'MercuryBowShock:TimeOrder', ...
        '%s timestamps are not strictly increasing.',files(k).name);

    day = datetime(extractBetween(string(files(k).name),1,8), ...
        'InputFormat','yyyyMMdd','TimeZone','UTC');
    dayStart = datenum(day);
    dayEnd = datenum(day+days(1));
    natural = raw(:,1)>=dayStart & raw(:,1)<dayEnd;
    d = raw(natural,:);
    assert(~isempty(d),'MercuryBowShock:EmptyDay', ...
        '%s has no samples inside its natural UTC day.',files(k).name);
    if d(1,1) <= previousEnd
        boundaryOverlapCount = boundaryOverlapCount + 1;
    end
    previousEnd = d(end,1);
    totalRows = totalRows + size(raw,1);

    Bmag = vecnorm(d(:,5:7),2,2);
    [qualityMask,dayExclusions] = mercury_bowshock_quality_mask(d);
    if ~isempty(dayExclusions)
        dayExclusions = addvars(dayExclusions, ...
            repmat(string(files(k).name),height(dayExclusions),1), ...
            'Before','StartUTC','NewVariableNames','SourceFile');
        exclusions = [exclusions; dayExclusions]; %#ok<AGROW>
    end
    [candidates,nCandidateExcluded] = seedCandidates( ...
        d,Bmag,files(k).name,qualityMask);
    candidateExcludedCount = candidateExcludedCount + nCandidateExcluded;
    crossings = [crossings; candidates]; %#ok<AGROW>

    metadata.File(k) = string(files(k).name);
    metadata.Bytes(k) = files(k).bytes;
    metadata.Modified(k) = datetime(files(k).datenum,'ConvertFrom','datenum');
    metadata.Rows(k) = size(raw,1);
    metadata.FirstUTC(k) = datetime(d(1,1),'ConvertFrom','datenum','TimeZone','UTC');
    metadata.LastUTC(k) = datetime(d(end,1),'ConvertFrom','datenum','TimeZone','UTC');
    metadata.ExtremeCount(k) = nnz(Bmag>1000);
    metadata.ExcludedCount(k) = nnz(qualityMask);
    metadata.InternalGapCount(k) = nnz(dt>1.5);
end

assert(totalRows == 2678493,'MercuryBowShock:RowCount', ...
    'Expected 2,678,493 source rows, found %d.',totalRows);
crossings = sortrows(crossings,'UTC');
crossings.ID = (1:height(crossings)).';
crossings = movevars(crossings,'ID','Before','UTC');

createdUTC = datetime('now','TimeZone','UTC');
validation = table(numel(files),totalRows,sum(metadata.ExtremeCount), ...
    sum(metadata.ExcludedCount),height(exclusions),candidateExcludedCount, ...
    sum(metadata.InternalGapCount),boundaryOverlapCount,height(crossings), ...
    'VariableNames',{'FileCount','SourceRows','ExtremeCount', ...
    'ExcludedSampleCount','ExcludedIntervalCount','CandidateExcludedCount', ...
    'InternalGapCount','BoundaryOverlapCount','AutoCandidateCount'});
save(fullfile(outputDir,'bowshock_crossings.mat'), ...
    'crossings','exclusions','metadata','validation','createdUTC');
writetable(crossings,fullfile(outputDir,'bowshock_crossings.csv'));
writetable(exclusions,fullfile(outputDir,'bowshock_exclusions.csv'));

[fitResult,fitCurve,fitSelection] = mercury_bowshock_fit(crossings,true);
writetable(fitResult,fullfile(outputDir,'bowshock_fit_results.csv'));
writetable(fitSelection,fullfile(outputDir,'bowshock_fit_selection.csv'));

exportDailyField(dataDir,outputDir,datetime(2013,12,1,'TimeZone','UTC'),crossings);
exportQualityExample(dataDir,outputDir,crossings);
exportICME(dataDir,outputDir,crossings);
exportFit(outputDir,crossings,fitResult,fitCurve,fitSelection);

result = struct('DataDir',string(dataDir),'OutputDir',string(outputDir), ...
    'Crossings',crossings,'Metadata',metadata,'Validation',validation, ...
    'Exclusions',exclusions,'FitResult',fitResult,'FitCurve',fitCurve, ...
    'FitSelection',fitSelection);
fprintf(['Mercury bow-shock batch complete: %d files, %d rows, %d candidates, ' ...
    '%d samples above 1000 nT, %d samples excluded with padding.\n'], ...
    numel(files),totalRows,height(crossings),sum(metadata.ExtremeCount), ...
    sum(metadata.ExcludedCount));
end

function [crossings,nExcluded] = seedCandidates(d,Bmag,sourceFile,qualityMask)
RM = 2440;
angle = deg2rad(7);
x = (d(:,2)*cos(angle)-d(:,3)*sin(angle))/RM;
y = (d(:,2)*sin(angle)+d(:,3)*cos(angle))/RM;
z = d(:,4)/RM;
rho = hypot(y,z-0.196);
p = [0.5,1.04,2.86];
surface = hypot(x-p(1),rho)+p(2)*(x-p(1))-p(3);
idx = find(surface(1:end-1).*surface(2:end)<=0 & ...
    surface(1:end-1)~=surface(2:end));

crossings = emptyCrossingTable();
nExcluded = 0;
for j = 1:numel(idx)
    i = idx(j);
    if qualityMask(i) || qualityMask(i+1)
        nExcluded = nExcluded + 1;
        continue
    end
    f = surface(i:i+1);
    alpha = abs(f(1))/(abs(f(1))+abs(f(2)));
    v = (1-alpha)*d(i,:) + alpha*d(i+1,:);
    bx = (v(2)*cos(angle)-v(3)*sin(angle))/RM;
    by = (v(2)*sin(angle)+v(3)*cos(angle))/RM;
    bz = v(4)/RM;
    if surface(i)>0 && surface(i+1)<0
        direction = "Inbound";
    else
        direction = "Outbound";
    end
    confidence = magneticConfidence(d,Bmag,i,direction,qualityMask);
    utc = datetime(v(1),'ConvertFrom','datenum','TimeZone','UTC');
    icme = utc>=datetime(2013,12,27,4,14,0,'TimeZone','UTC') && ...
        utc<=datetime(2013,12,27,15,27,0,'TimeZone','UTC');
    row = table(utc,direction,v(2),v(3),v(4),bx,by,bz, ...
        hypot(by,bz-0.196),"Auto",confidence,icme,string(sourceFile),i, ...
        'VariableNames',crossings.Properties.VariableNames);
    crossings = [crossings; row]; %#ok<AGROW>
end
end

function confidence = magneticConfidence(d,Bmag,i,direction,qualityMask)
% Score is diagnostic only: the Winslow intersection supplies the location.
n = size(d,1);
pre = max(1,i-120):max(1,i-1);
post = min(n,i+1):min(n,i+120);
pre = pre(~qualityMask(pre));
post = post(~qualityMask(post));
if numel(pre)<20 || numel(post)<20
    confidence = 0.25;
    return
end
preVar = median(vecnorm(d(pre,5:7)-median(d(pre,5:7),1),2,2));
postVar = median(vecnorm(d(post,5:7)-median(d(post,5:7),1),2,2));
preB = median(Bmag(pre));
postB = median(Bmag(post));
if direction=="Inbound"
    evidence = (postVar-preVar)+0.20*(postB-preB);
else
    evidence = (preVar-postVar)+0.20*(preB-postB);
end
confidence = max(0.05,min(0.99,1/(1+exp(-evidence/8))));
end

function t = emptyCrossingTable()
t = table('Size',[0 14], ...
    'VariableTypes',{'datetime','string','double','double','double','double', ...
    'double','double','double','string','double','logical','string','double'}, ...
    'VariableNames',{'UTC','Direction','X_MSO_km','Y_MSO_km','Z_MSO_km', ...
    'X_ab_RM','Y_ab_RM','Z_MSO_RM','Rho_RM','Status','Confidence', ...
    'ICMEFlag','SourceFile','SampleIndex'});
t.UTC.TimeZone = 'UTC';
end

function t = emptyExclusionTable()
t = table('Size',[0 7], ...
    'VariableTypes',{'string','datetime','datetime','double','double', ...
    'double','string'}, ...
    'VariableNames',{'SourceFile','StartUTC','EndUTC','ExcludedSamples', ...
    'ExtremeSamples','MaxB_nT','Reason'});
t.StartUTC.TimeZone = 'UTC';
t.EndUTC.TimeZone = 'UTC';
end

function [d,t,Bmag,qualityMask,intervals] = loadNaturalDay(dataDir,day)
name = sprintf('%s_01s.mat',datestr(day,'yyyymmdd'));
loaded = load(fullfile(dataDir,name),'data_mso');
startNum = datenum(day);
d = loaded.data_mso(loaded.data_mso(:,1)>=startNum & ...
    loaded.data_mso(:,1)<startNum+1,:);
t = datetime(d(:,1),'ConvertFrom','datenum','TimeZone','UTC');
Bmag = vecnorm(d(:,5:7),2,2);
[qualityMask,intervals] = mercury_bowshock_quality_mask(d);
end

function exportDailyField(dataDir,outputDir,day,crossings)
[d,t,Bmag,qualityMask,intervals] = loadNaturalDay(dataDir,day);
displayValues = [d(:,5:7),Bmag];
displayValues(qualityMask,:) = NaN;
fig = figure('Visible','off','Color','w','Position',[50 50 1600 850]);
theme(fig,'light');
ax = axes(fig,'Position',[0.08 0.13 0.87 0.78]);
styleAxes(ax);
plot(ax,t,displayValues(:,1),'Color',[0.10 0.35 0.85], ...
    'DisplayName','B_x'); hold(ax,'on');
plot(ax,t,displayValues(:,2),'Color',[0.85 0.20 0.15],'DisplayName','B_y');
plot(ax,t,displayValues(:,3),'Color',[0.15 0.65 0.25],'DisplayName','B_z');
plot(ax,t,displayValues(:,4),'k','LineWidth',1,'DisplayName','|B|');
markExclusions(ax,intervals,true);
markCrossings(ax,crossings,day);
formatTimeAxis(ax,day); ylabel(ax,'Magnetic field (nT)');
title(ax,'MESSENGER magnetic field — 2013-12-01 (PPT page 8 style)');
lgd = legend(ax,'Location','northoutside','Orientation','horizontal');
styleLegend(lgd);
robustFieldLimits(ax,[d(:,5:7),Bmag]);
exportgraphics(fig,fullfile(outputDir,'page08_daily_field_20131201.png'), ...
    'Resolution',300);
close(fig);
end

function exportQualityExample(dataDir,outputDir,crossings)
day = datetime(2013,12,4,'TimeZone','UTC');
[d,t,Bmag,qualityMask,intervals] = loadNaturalDay(dataDir,day);
values = [d(:,5:7),Bmag];
displayValues = values;
displayValues(qualityMask,:) = NaN;

fig = figure('Visible','off','Color','w','Position',[50 50 1600 850]);
theme(fig,'light');
ax = axes(fig,'Position',[0.08 0.14 0.88 0.76]);
styleAxes(ax);
colors = [0.10 0.35 0.85;0.85 0.20 0.15;0.15 0.65 0.25;0 0 0];
labels = {'B_x','B_y','B_z','|B|'};
for j = 1:4
    plot(ax,t,displayValues(:,j),'Color',colors(j,:), ...
        'DisplayName',labels{j}); hold(ax,'on');
end
markExclusions(ax,intervals,true);
markCrossings(ax,crossings,day);
formatTimeAxis(ax,day);
ylabel(ax,'Magnetic field (nT)');
title(ax,{ ...
    'Data-quality exclusion example — 2013-12-04', ...
    sprintf(['Source values retained; %d samples masked in plots ' ...
    'and candidate scoring'],nnz(qualityMask))});
lgd = legend(ax,'Location','northoutside','Orientation','horizontal');
styleLegend(lgd);
robustFieldLimits(ax,values);
exportgraphics(fig, ...
    fullfile(outputDir,'quality_artifact_mask_20131204.png'), ...
    'Resolution',300);
close(fig);
end

function exportICME(dataDir,outputDir,crossings)
day = datetime(2013,12,27,'TimeZone','UTC');
[d,t,Bmag,qualityMask,intervals] = loadNaturalDay(dataDir,day);
labels = {'B_x','B_y','B_z','|B|'};
colors = [0.10 0.35 0.85;0.85 0.20 0.15;0.15 0.65 0.25;0 0 0];
values = [d(:,5:7),Bmag];
displayValues = values;
displayValues(qualityMask,:) = NaN;
events = icmeEvents();

fig = figure('Visible','off','Color','w','Position',[30 30 1600 1100]);
theme(fig,'light');
tl = tiledlayout(fig,4,1,'TileSpacing','compact','Padding','compact');
for j = 1:4
    ax = nexttile(tl);
    styleAxes(ax);
    plot(ax,t,displayValues(:,j),'Color',colors(j,:),'LineWidth',0.8); hold(ax,'on');
    markExclusions(ax,intervals,j==1);
    markEvents(ax,events,j==1);
    markCrossings(ax,crossings,day);
    ylabel(ax,sprintf('%s (nT)',labels{j}));
    robustFieldLimits(ax,values(:,j));
    grid(ax,'on');
    if j<4, ax.XTickLabel=[]; end
end
xlabel(tl,'UTC'); title(tl,'MESSENGER 2013-12-27 ICME and bow-shock crossings');
exportgraphics(fig,fullfile(outputDir,'page13_icme_20131227_stacked.png'), ...
    'Resolution',300);
close(fig);

fig = figure('Visible','off','Color','w','Position',[50 50 1600 850]);
theme(fig,'light');
ax = axes(fig,'Position',[0.08 0.14 0.88 0.76]);
styleAxes(ax);
for j = 1:4
    plot(ax,t,displayValues(:,j),'Color',colors(j,:), ...
        'DisplayName',labels{j}); hold(ax,'on');
end
markExclusions(ax,intervals,true);
markEvents(ax,events,true); markCrossings(ax,crossings,day);
formatTimeAxis(ax,day); ylabel(ax,'Magnetic field (nT)');
title(ax,'2013-12-27 ICME overview (PPT page 14 style)');
lgd = legend(ax,'Location','northoutside','Orientation','horizontal');
styleLegend(lgd);
robustFieldLimits(ax,values);
exportgraphics(fig,fullfile(outputDir,'page14_icme_20131227_overview.png'), ...
    'Resolution',300);
close(fig);
end

function exportFit(outputDir,crossings,fitResult,curve,fitSelection)
selectedIDs = fitSelection.ID(fitSelection.SelectedForFit);
use = ismember(crossings.ID,selectedIDs);
x = crossings.X_ab_RM(use);
rho = hypot(crossings.Y_ab_RM(use),crossings.Z_MSO_RM(use)-0.196);
theta = linspace(-2.55,2.55,1200);
ref = [0.5,1.04,2.86];
r = ref(3)./(1+ref(2)*cos(theta));
xRef = ref(1)+r.*cos(theta); rhoRef = abs(r.*sin(theta));
valid = xRef>-6 & xRef<4 & rhoRef<8 & isfinite(xRef);

fig = figure('Visible','off','Color','w','Position',[100 50 1200 950]);
theme(fig,'light');
ax = axes(fig); hold(ax,'on');
styleAxes(ax);
fill(ax,cos(linspace(0,2*pi,400)),sin(linspace(0,2*pi,400)), ...
    [0.72 0.72 0.72],'EdgeColor',[0.25 0.25 0.25], ...
    'DisplayName','Mercury (1 R_M)');
plot(ax,xRef(valid),rhoRef(valid),'Color',[0.92 0.65 0.05], ...
    'LineWidth',2.5,'DisplayName','Winslow et al. (2013)');
plot(ax,curve.X_ab_RM,curve.rho_RM,'r-','LineWidth',2.2, ...
    'DisplayName','This fit');
scatter(ax,x,rho,25,'k','filled','MarkerFaceAlpha',0.65, ...
    'DisplayName','Crossings');
axis(ax,'equal'); xlim(ax,[-4 3]); ylim(ax,[0 5]);
grid(ax,'on'); xlabel(ax,'X''_{MSO} (R_M)');
ylabel(ax,'\rho = [Y''^2+(Z-0.196)^2]^{1/2} (R_M)');
title(ax,sprintf(['Mercury bow-shock conic fit — %s\n' ...
    'X_0=%.3f, \\epsilon=%.3f, L=%.3f, L_{SSP}=%.3f R_M'], ...
    fitResult.Mode,fitResult.X0,fitResult.epsilon,fitResult.L,fitResult.L_SSP_RM));
lgd = legend(ax,'Location','northeast');
styleLegend(lgd);
text(ax,-3.85,4.65,sprintf('N=%d, RMSE=%.3f R_M (%.0f km)', ...
    fitResult.NPoints,fitResult.RMSE_RM,fitResult.RMSE_km), ...
    'BackgroundColor','w','Margin',5);
exportgraphics(fig,fullfile(outputDir,'page17_bowshock_fit.png'), ...
    'Resolution',300);
close(fig);
end

function markExclusions(ax,intervals,showLabel)
for k = 1:height(intervals)
    xline(ax,intervals.StartUTC(k),'-.','Color',[0.35 0.35 0.35], ...
        'LineWidth',1.2,'HandleVisibility','off');
    if showLabel
        label = 'excluded artifact';
    else
        label = '';
    end
    xline(ax,intervals.EndUTC(k),'-.',label,'Color',[0.35 0.35 0.35], ...
        'LineWidth',1.2,'LabelVerticalAlignment','bottom', ...
        'HandleVisibility','off');
end
end

function markCrossings(ax,crossings,day)
mask = dateshift(crossings.UTC,'start','day')==dateshift(day,'start','day') & ...
    string(crossings.Status)~="Rejected";
rows = crossings(mask,:);
for k = 1:height(rows)
    if rows.Direction(k)=="Inbound"
        c = [0.55 0.15 0.75];
    else
        c = [0.05 0.60 0.65];
    end
    xline(ax,rows.UTC(k),':','Color',c,'LineWidth',0.8, ...
        'HandleVisibility','off');
end
end

function events = icmeEvents()
events.Time = [datetime(2013,12,27,4,14,0,'TimeZone','UTC'); ...
    datetime(2013,12,27,5,36,0,'TimeZone','UTC'); ...
    datetime(2013,12,27,15,27,0,'TimeZone','UTC')];
events.Label = ["Interplanetary shock";"Magnetic ejecta start"; ...
    "Magnetic ejecta end"];
events.Color = [0.80 0.10 0.10;0.10 0.45 0.80;0.10 0.45 0.80];
end

function markEvents(ax,events,showLabels)
for k = 1:numel(events.Time)
    if showLabels
        label = events.Label(k);
    else
        label = "";
    end
    xline(ax,events.Time(k),'--',label,'Color',events.Color(k,:), ...
        'LineWidth',1.4,'LabelVerticalAlignment','middle', ...
        'LabelOrientation','horizontal','HandleVisibility','off');
end
end

function formatTimeAxis(ax,day)
xlim(ax,[day day+days(1)]);
xticks(ax,day+hours(0:3:24));
xtickformat(ax,'HH:mm');
xlabel(ax,sprintf('UTC on %s',datestr(day,'yyyy-mm-dd')));
grid(ax,'on');
end

function robustFieldLimits(ax,values)
v = values(isfinite(values) & abs(values)<=1000);
if isempty(v), return; end
lo = prctile(v,0.25); hi = prctile(v,99.75);
span = max(hi-lo,10);
ylim(ax,[lo-0.08*span,hi+0.08*span]);
end

function styleAxes(ax)
set(ax,'Color','w','XColor','k','YColor','k', ...
    'GridColor',[0.72 0.72 0.72],'MinorGridColor',[0.86 0.86 0.86]);
end

function styleLegend(lgd)
set(lgd,'Color','w','TextColor','k','EdgeColor',[0.35 0.35 0.35]);
end
