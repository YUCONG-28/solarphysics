function app = mercury_bowshock_app(dataDir, outputDir)
%MERCURY_BOWSHOCK_APP Interactive review of MESSENGER bow-shock crossings.
%   MERCURY_BOWSHOCK_APP() locates 201312_01s beside this analysis folder.
%   The source MAT files are loaded read-only. Use the toolbar for horizontal
%   zoom, pan, and data tips; use the controls to review crossing candidates.

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

stateFile = fullfile(outputDir,'bowshock_crossings.mat');
if ~isfile(stateFile)
    batch = mercury_bowshock_batch(dataDir,outputDir);
    crossings = batch.Crossings;
    exclusions = batch.Exclusions;
else
    saved = load(stateFile);
    crossings = saved.crossings;
    if isfield(saved,'exclusions')
        exclusions = saved.exclusions;
    else
        exclusions = table();
    end
end
crossings = normalizeCrossings(crossings);

daysList = datetime(2013,12,1:31,'TimeZone','UTC');
dayLabels = cellstr(string(daysList,'yyyy-MM-dd'));
state = struct('DataDir',dataDir,'OutputDir',outputDir, ...
    'Crossings',crossings,'Exclusions',exclusions, ...
    'Days',daysList,'DayIndex',1, ...
    'SelectedID',NaN,'Axes',gobjects(0),'PlotMode',"Overlay", ...
    'ClickMode',"None",'Data',[],'Time',[],'Bmag',[], ...
    'QualityMask',[],'DayExclusions',table());

fig = figure('Name','Mercury Bow-Shock Review — December 2013', ...
    'NumberTitle','off','Color',[0.96 0.96 0.96], ...
    'Position',[40 40 1600 950],'ToolBar','figure', ...
    'CloseRequestFcn',@closeApp);
theme(fig,'light');
state.Figure = fig;

uicontrol(fig,'Style','text','String','Date (UTC)', ...
    'Units','normalized','Position',[0.012 0.952 0.055 0.026], ...
    'BackgroundColor',fig.Color,'HorizontalAlignment','left');
state.DatePopup = uicontrol(fig,'Style','popupmenu','String',dayLabels, ...
    'Value',1,'Units','normalized','Position',[0.066 0.951 0.095 0.033], ...
    'Callback',@changeDay);
uicontrol(fig,'Style','text','String','View', ...
    'Units','normalized','Position',[0.169 0.952 0.035 0.026], ...
    'BackgroundColor',fig.Color,'HorizontalAlignment','left');
state.ViewPopup = uicontrol(fig,'Style','popupmenu', ...
    'String',{'Overlay','Four stacked panels'},'Value',1, ...
    'Units','normalized','Position',[0.202 0.951 0.115 0.033], ...
    'Callback',@changeView);

makeButton('Previous',[0.326 0.951 0.055 0.034],@previousCandidate);
makeButton('Next',[0.384 0.951 0.045 0.034],@nextCandidate);
makeButton('Add / Move',[0.438 0.951 0.070 0.034],@armAddMove);
makeButton('Confirm',[0.511 0.951 0.055 0.034],@confirmCandidate);
makeButton('Reject',[0.569 0.951 0.050 0.034],@rejectCandidate);
makeButton('Delete',[0.622 0.951 0.050 0.034],@deleteCandidate);
makeButton('Save',[0.681 0.951 0.045 0.034],@saveReview);
makeButton('Final fit / Export',[0.729 0.951 0.092 0.034],@finalFit);
makeButton('Reset zoom',[0.824 0.951 0.068 0.034],@resetZoom);
state.HelpText = uicontrol(fig,'Style','text', ...
    'String','Toolbar: horizontal zoom, pan, data tips', ...
    'Units','normalized','Position',[0.012 0.918 0.53 0.027], ...
    'BackgroundColor',fig.Color,'HorizontalAlignment','left', ...
    'ForegroundColor',[0.2 0.2 0.2]);
state.StatusText = uicontrol(fig,'Style','text','String','Ready', ...
    'Units','normalized','Position',[0.545 0.914 0.347 0.033], ...
    'BackgroundColor',[0.90 0.93 0.98], ...
    'HorizontalAlignment','left','FontWeight','bold');

state.Table = uitable(fig,'Units','normalized', ...
    'Position',[0.805 0.080 0.187 0.820], ...
    'ColumnName',{'UTC','Dir','Status','Confidence'}, ...
    'ColumnWidth',{74 58 66 64},'RowName',[], ...
    'CellSelectionCallback',@selectTableRow);
state.InfoText = uicontrol(fig,'Style','text','Units','normalized', ...
    'Position',[0.012 0.012 0.98 0.047], ...
    'BackgroundColor',[1 1 1],'HorizontalAlignment','left', ...
    'String',['Auto candidates are Winslow-surface intersections. ' ...
    'Only Confirmed rows enter the final fit.']);

guidata(fig,state);
installHorizontalExploration(fig);
loadAndRenderDay(fig);
if nargout
    app = fig;
end

    function makeButton(label,pos,callback)
        uicontrol(fig,'Style','pushbutton','String',label, ...
            'Units','normalized','Position',pos,'Callback',callback);
    end

    function changeDay(src,~)
        s = guidata(fig);
        s.DayIndex = src.Value;
        s.SelectedID = NaN;
        s.ClickMode = "None";
        guidata(fig,s);
        loadAndRenderDay(fig);
    end

    function changeView(src,~)
        s = guidata(fig);
        if src.Value==1
            s.PlotMode = "Overlay";
        else
            s.PlotMode = "Stacked";
        end
        guidata(fig,s);
        renderDay(fig);
    end

    function previousCandidate(~,~)
        stepCandidate(-1);
    end

    function nextCandidate(~,~)
        stepCandidate(1);
    end

    function stepCandidate(delta)
        s = guidata(fig);
        ids = currentDayIDs(s);
        if isempty(ids), return; end
        pos = find(ids==s.SelectedID,1);
        if isempty(pos)
            pos = 1;
        else
            pos = mod(pos-1+delta,numel(ids))+1;
        end
        s.SelectedID = ids(pos);
        guidata(fig,s);
        renderDay(fig);
        centerSelected(fig);
    end

    function armAddMove(~,~)
        s = guidata(fig);
        s.ClickMode = "AddMove";
        if isfinite(s.SelectedID)
            instruction = 'Click a field panel to move the selected crossing.';
        else
            instruction = 'Click a field panel to add a manual crossing.';
        end
        s.StatusText.String = instruction;
        fig.Pointer = 'crosshair';
        fig.WindowButtonDownFcn = @placeCandidate;
        guidata(fig,s);
    end

    function placeCandidate(~,~)
        s = guidata(fig);
        hitAx = ancestor(hittest(fig),'axes');
        if isempty(hitAx) || ~any(hitAx==s.Axes)
            return
        end
        figurePoint = fig.CurrentPoint;
        axesPixels = getpixelposition(hitAx,true);
        fraction = (figurePoint(1)-axesPixels(1))/axesPixels(3);
        fraction = max(0,min(1,fraction));
        limits = xlim(hitAx);
        clickTime = limits(1) + fraction*(limits(2)-limits(1));
        [~,sample] = min(abs(seconds(s.Time-clickTime)));
        utc = s.Time(sample);
        if s.QualityMask(sample)
            s.StatusText.String = ['This interval is an excluded periodic ' ...
                'artifact; no crossing was added or moved.'];
            s.ClickMode = "None";
            fig.Pointer = 'arrow';
            fig.WindowButtonDownFcn = '';
            guidata(fig,s);
            return
        end

        id = s.SelectedID;
        if isfinite(id)
            row = find(s.Crossings.ID==id,1);
            s.Crossings = updateCrossingFromSample(s.Crossings,row,s.Data, ...
                s.Bmag,s.QualityMask,sample,"Manual");
            message = sprintf('Moved crossing %d to %s UTC.',id, ...
                datestr(utc,'yyyy-mm-dd HH:MM:SS'));
        else
            newID = max([s.Crossings.ID;0])+1;
            newRow = makeManualCrossing( ...
                s.Data,s.Bmag,s.QualityMask,sample,newID);
            s.Crossings = [s.Crossings; newRow];
            s.Crossings = sortrows(s.Crossings,'UTC');
            s.SelectedID = newID;
            message = sprintf('Added manual crossing %d at %s UTC.',newID, ...
                datestr(utc,'yyyy-mm-dd HH:MM:SS'));
        end
        s.ClickMode = "None";
        s.StatusText.String = message;
        fig.Pointer = 'arrow';
        fig.WindowButtonDownFcn = '';
        guidata(fig,s);
        renderDay(fig);
    end

    function confirmCandidate(~,~)
        setSelectedStatus("Confirmed");
    end

    function rejectCandidate(~,~)
        setSelectedStatus("Rejected");
    end

    function setSelectedStatus(status)
        s = guidata(fig);
        if ~isfinite(s.SelectedID)
            s.StatusText.String = 'Select a candidate first.';
            guidata(fig,s); return
        end
        row = find(s.Crossings.ID==s.SelectedID,1);
        if status=="Confirmed" && ...
                isTimeExcluded(s.Crossings.UTC(row),s.Exclusions)
            s.Crossings.Status(row) = "Rejected";
            s.StatusText.String = sprintf(['Crossing %d lies in an excluded ' ...
                'artifact interval and was rejected.'],s.SelectedID);
            guidata(fig,s);
            renderDay(fig);
            return
        end
        s.Crossings.Status(row) = status;
        s.StatusText.String = sprintf('Crossing %d set to %s.', ...
            s.SelectedID,status);
        guidata(fig,s);
        renderDay(fig);
    end

    function deleteCandidate(~,~)
        s = guidata(fig);
        if ~isfinite(s.SelectedID)
            s.StatusText.String = 'Select a candidate first.';
            guidata(fig,s); return
        end
        id = s.SelectedID;
        s.Crossings(s.Crossings.ID==id,:) = [];
        s.SelectedID = NaN;
        s.StatusText.String = sprintf('Deleted crossing %d (save to persist).',id);
        guidata(fig,s);
        renderDay(fig);
    end

    function selectTableRow(~,event)
        if isempty(event.Indices), return; end
        s = guidata(fig);
        ids = currentDayIDs(s);
        row = event.Indices(1);
        if row<=numel(ids)
            s.SelectedID = ids(row);
            guidata(fig,s);
            renderDay(fig);
            centerSelected(fig);
        end
    end

    function saveReview(~,~)
        s = guidata(fig);
        saveCrossings(s);
        s.StatusText.String = sprintf('Saved %d rows (%d confirmed).', ...
            height(s.Crossings),nnz(string(s.Crossings.Status)=="Confirmed"));
        guidata(fig,s);
    end

    function finalFit(~,~)
        s = guidata(fig);
        bad = isTimeExcluded(s.Crossings.UTC,s.Exclusions);
        s.Crossings.Status(bad & ...
            string(s.Crossings.Status)=="Confirmed") = "Rejected";
        confirmed = string(s.Crossings.Status)=="Confirmed";
        if nnz(confirmed)<3
            s.StatusText.String = sprintf(['Final fit needs at least 3 Confirmed ' ...
                'points; currently %d.'],nnz(confirmed));
            guidata(fig,s); return
        end
        saveCrossings(s);
        [fitResult,curve,selection] = mercury_bowshock_fit( ...
            s.Crossings,false);
        writetable(fitResult,fullfile(s.OutputDir,'bowshock_fit_results.csv'));
        writetable(selection,fullfile(s.OutputDir,'bowshock_fit_selection.csv'));
        exportInteractiveFit(s.Crossings,fitResult,curve,selection,s.OutputDir);
        s.StatusText.String = sprintf(['Final fit exported: N=%d, ' ...
            'discarded inner=%d, L_{SSP}=%.3f R_M, RMSE=%.3f R_M.'], ...
            fitResult.NPoints,fitResult.NDiscardedInner, ...
            fitResult.L_SSP_RM,fitResult.RMSE_RM);
        guidata(fig,s);
    end

    function resetZoom(~,~)
        s = guidata(fig);
        day = s.Days(s.DayIndex);
        set(s.Axes,'XLim',[day day+days(1)]);
    end

    function closeApp(~,~)
        if isgraphics(fig)
            delete(fig);
        end
    end
end

function loadAndRenderDay(fig)
s = guidata(fig);
day = s.Days(s.DayIndex);
file = fullfile(s.DataDir,sprintf('%s_01s.mat',datestr(day,'yyyymmdd')));
loaded = load(file,'data_mso');
mask = loaded.data_mso(:,1)>=datenum(day) & ...
    loaded.data_mso(:,1)<datenum(day)+1;
s.Data = loaded.data_mso(mask,:);
s.Time = datetime(s.Data(:,1),'ConvertFrom','datenum','TimeZone','UTC');
s.Bmag = vecnorm(s.Data(:,5:7),2,2);
[s.QualityMask,s.DayExclusions] = ...
    mercury_bowshock_quality_mask(s.Data);
s.StatusText.String = sprintf('Loaded %s: %d samples.', ...
    datestr(day,'yyyy-mm-dd'),size(s.Data,1));
guidata(fig,s);
renderDay(fig);
end

function renderDay(fig)
s = guidata(fig);
delete(s.Axes(isgraphics(s.Axes)));
s.Axes = gobjects(0);
day = s.Days(s.DayIndex);
colors = [0.10 0.35 0.85;0.85 0.20 0.15;0.15 0.65 0.25;0 0 0];
labels = {'B_x','B_y','B_z','|B|'};
values = [s.Data(:,5:7),s.Bmag];
displayValues = values;
displayValues(s.QualityMask,:) = NaN;

if s.PlotMode=="Overlay"
    ax = axes(fig,'Position',[0.060 0.115 0.720 0.775]);
    set(ax,'Color','w','XColor','k','YColor','k', ...
        'GridColor',[0.72 0.72 0.72]);
    hold(ax,'on');
    for j = 1:4
        plot(ax,s.Time,displayValues(:,j),'Color',colors(j,:), ...
            'DisplayName',labels{j});
    end
    ylabel(ax,'Magnetic field (nT)');
    legend(ax,'Location','northoutside','Orientation','horizontal');
    s.Axes = ax;
else
    for j = 1:4
        bottom = 0.115+(4-j)*0.190;
        ax = axes(fig,'Position',[0.060 bottom 0.720 0.172]);
        set(ax,'Color','w','XColor','k','YColor','k', ...
            'GridColor',[0.72 0.72 0.72]);
        plot(ax,s.Time,displayValues(:,j),'Color',colors(j,:)); hold(ax,'on');
        ylabel(ax,sprintf('%s (nT)',labels{j}));
        if j<4, ax.XTickLabel=[]; end
        s.Axes(end+1) = ax; %#ok<AGROW>
    end
end

for ax = reshape(s.Axes,1,[])
    grid(ax,'on');
    xlim(ax,[day day+days(1)]);
    xticks(ax,day+hours(0:3:24));
    xtickformat(ax,'HH:mm');
    robustLimits(ax,values);
    drawExclusionLines(ax,s.DayExclusions,ax==s.Axes(1));
    drawCandidateLines(ax,s);
    if day==datetime(2013,12,27,'TimeZone','UTC')
        drawICMELines(ax);
    end
end
xlabel(s.Axes(end),sprintf('UTC on %s',datestr(day,'yyyy-mm-dd')));
title(s.Axes(1),sprintf(['MESSENGER MSO magnetic field — %s | ' ...
    '1 s data; periodic artifacts masked, source MAT unchanged'], ...
    datestr(day,'yyyy-mm-dd')));

s = updateTable(s);
guidata(fig,s);
end

function drawCandidateLines(ax,s)
day = s.Days(s.DayIndex);
mask = dateshift(s.Crossings.UTC,'start','day')==day;
rows = s.Crossings(mask,:);
for k = 1:height(rows)
    selected = rows.ID(k)==s.SelectedID;
    switch string(rows.Status(k))
        case "Confirmed"
            color = [0.05 0.62 0.18]; style = '-'; width = 1.5;
        case "Rejected"
            color = [0.75 0.15 0.15]; style = ':'; width = 0.8;
        otherwise
            color = [0.45 0.22 0.70]; style = '--'; width = 0.9;
    end
    if selected
        color = [1.0 0.25 0.0]; width = 2.5;
    end
    xline(ax,rows.UTC(k),style,'Color',color,'LineWidth',width, ...
        'HitTest','off','HandleVisibility','off');
end
end

function drawICMELines(ax)
times = [datetime(2013,12,27,4,14,0,'TimeZone','UTC'), ...
    datetime(2013,12,27,5,36,0,'TimeZone','UTC'), ...
    datetime(2013,12,27,15,27,0,'TimeZone','UTC')];
labels = ["IP shock","ME start","ME end"];
for k = 1:3
    xline(ax,times(k),'-.',labels(k),'Color',[0.05 0.45 0.80], ...
        'LineWidth',1.3,'LabelVerticalAlignment','middle', ...
        'HandleVisibility','off','HitTest','off');
end
end

function s = updateTable(s)
day = s.Days(s.DayIndex);
mask = dateshift(s.Crossings.UTC,'start','day')==day;
rows = s.Crossings(mask,:);
data = cell(height(rows),4);
for k = 1:height(rows)
    data{k,1} = datestr(rows.UTC(k),'HH:MM:SS');
    data{k,2} = char(rows.Direction(k));
    data{k,3} = char(rows.Status(k));
    data{k,4} = sprintf('%.2f',rows.Confidence(k));
end
s.Table.Data = data;
s.InfoText.String = sprintf(['Day candidates: %d | Confirmed: %d | ' ...
    'Auto: %d | Rejected: %d | Excluded samples: %d'], ...
    height(rows),nnz(string(rows.Status)=="Confirmed"), ...
    nnz(string(rows.Status)=="Auto"),nnz(string(rows.Status)=="Rejected"), ...
    nnz(s.QualityMask));
end

function drawExclusionLines(ax,intervals,showLabel)
for k = 1:height(intervals)
    xline(ax,intervals.StartUTC(k),'-.','Color',[0.35 0.35 0.35], ...
        'LineWidth',1.2,'HitTest','off','HandleVisibility','off');
    if showLabel
        label = 'excluded artifact';
    else
        label = '';
    end
    xline(ax,intervals.EndUTC(k),'-.',label, ...
        'Color',[0.35 0.35 0.35],'LineWidth',1.2, ...
        'LabelVerticalAlignment','bottom','HitTest','off', ...
        'HandleVisibility','off');
end
end

function ids = currentDayIDs(s)
day = s.Days(s.DayIndex);
mask = dateshift(s.Crossings.UTC,'start','day')==day;
ids = s.Crossings.ID(mask).';
end

function centerSelected(fig)
s = guidata(fig);
row = find(s.Crossings.ID==s.SelectedID,1);
if isempty(row), return; end
t = s.Crossings.UTC(row);
for ax = reshape(s.Axes,1,[])
    xlim(ax,[t-minutes(12),t+minutes(12)]);
end
end

function crossings = updateCrossingFromSample( ...
        crossings,row,d,Bmag,qualityMask,sample,status)
RM = 2440; angle = deg2rad(7);
xab = (d(sample,2)*cos(angle)-d(sample,3)*sin(angle))/RM;
yab = (d(sample,2)*sin(angle)+d(sample,3)*cos(angle))/RM;
z = d(sample,4)/RM;
crossings.UTC(row) = datetime(d(sample,1),'ConvertFrom','datenum','TimeZone','UTC');
crossings.X_MSO_km(row) = d(sample,2);
crossings.Y_MSO_km(row) = d(sample,3);
crossings.Z_MSO_km(row) = d(sample,4);
crossings.X_ab_RM(row) = xab;
crossings.Y_ab_RM(row) = yab;
crossings.Z_MSO_RM(row) = z;
crossings.Rho_RM(row) = hypot(yab,z-0.196);
crossings.SampleIndex(row) = sample;
crossings.SourceFile(row) = ...
    string(sprintf('%s_01s.mat',datestr(crossings.UTC(row),'yyyymmdd')));
crossings.Status(row) = status;
crossings.Confidence(row) = localConfidence(d,Bmag,qualityMask,sample, ...
    crossings.Direction(row));
crossings.ICMEFlag(row) = inICME(crossings.UTC(row));
end

function row = makeManualCrossing(d,Bmag,qualityMask,sample,id)
direction = inferDirection(d,sample);
t = table(datetime(d(sample,1),'ConvertFrom','datenum','TimeZone','UTC'), ...
    direction,d(sample,2),d(sample,3),d(sample,4),0,0,0,0, ...
    "Manual",0,false,string(sprintf('%s_01s.mat', ...
    datestr(d(sample,1),'yyyymmdd'))),sample, ...
    'VariableNames',{'UTC','Direction','X_MSO_km','Y_MSO_km','Z_MSO_km', ...
    'X_ab_RM','Y_ab_RM','Z_MSO_RM','Rho_RM','Status','Confidence', ...
    'ICMEFlag','SourceFile','SampleIndex'});
t = addvars(t,id,'Before','UTC','NewVariableNames','ID');
t = updateCrossingFromSample(t,1,d,Bmag,qualityMask,sample,"Manual");
row = t;
end

function direction = inferDirection(d,sample)
r = vecnorm(d(:,2:4),2,2);
a = max(1,sample-30);
b = min(size(d,1),sample+30);
if r(b)<r(a)
    direction = "Inbound";
else
    direction = "Outbound";
end
end

function c = localConfidence(d,Bmag,qualityMask,i,direction)
n = size(d,1);
pre = max(1,i-120):max(1,i-1);
post = min(n,i+1):min(n,i+120);
pre = pre(~qualityMask(pre));
post = post(~qualityMask(post));
if numel(pre)<20 || numel(post)<20
    c = 0.25; return
end
pv = median(vecnorm(d(pre,5:7)-median(d(pre,5:7),1),2,2));
qv = median(vecnorm(d(post,5:7)-median(d(post,5:7),1),2,2));
pb = median(Bmag(pre)); qb = median(Bmag(post));
if string(direction)=="Inbound"
    evidence = qv-pv+0.2*(qb-pb);
else
    evidence = pv-qv+0.2*(pb-qb);
end
c = max(0.05,min(0.99,1/(1+exp(-evidence/8))));
end

function tf = inICME(t)
tf = t>=datetime(2013,12,27,4,14,0,'TimeZone','UTC') && ...
    t<=datetime(2013,12,27,15,27,0,'TimeZone','UTC');
end

function saveCrossings(s)
crossings = s.Crossings; %#ok<NASGU>
exclusions = s.Exclusions; %#ok<NASGU>
createdUTC = datetime('now','TimeZone','UTC'); %#ok<NASGU>
stateFile = fullfile(s.OutputDir,'bowshock_crossings.mat');
if isfile(stateFile)
    old = load(stateFile);
    if isfield(old,'metadata'), metadata = old.metadata; else, metadata=[]; end %#ok<NASGU>
    if isfield(old,'validation'), validation=old.validation; else, validation=[]; end %#ok<NASGU>
    save(stateFile,'crossings','exclusions','metadata','validation','createdUTC');
else
    save(stateFile,'crossings','exclusions','createdUTC');
end
writetable(crossings,fullfile(s.OutputDir,'bowshock_crossings.csv'));
if ~isempty(exclusions)
    writetable(exclusions,fullfile(s.OutputDir,'bowshock_exclusions.csv'));
end
end

function crossings = normalizeCrossings(crossings)
crossings.UTC.TimeZone = 'UTC';
crossings.Direction = string(crossings.Direction);
crossings.Status = string(crossings.Status);
crossings.SourceFile = string(crossings.SourceFile);
if ~ismember('ID',crossings.Properties.VariableNames)
    crossings.ID = (1:height(crossings)).';
    crossings = movevars(crossings,'ID','Before','UTC');
end
end

function installHorizontalExploration(fig)
z = zoom(fig); z.Motion = 'horizontal';
p = pan(fig); p.Motion = 'horizontal';
datacursormode(fig,'on');
datacursormode(fig,'off');
end

function robustLimits(ax,values)
v = values(isfinite(values) & abs(values)<=1000);
if isempty(v), return; end
lo = prctile(v,0.25); hi = prctile(v,99.75);
span = max(hi-lo,10);
ylim(ax,[lo-0.08*span,hi+0.08*span]);
end

function tf = isTimeExcluded(time,exclusions)
tf = false(size(time));
if isempty(exclusions) || ~ismember('StartUTC', ...
        exclusions.Properties.VariableNames)
    return
end
for k = 1:height(exclusions)
    tf = tf | (time>=exclusions.StartUTC(k) & time<=exclusions.EndUTC(k));
end
end

function exportInteractiveFit(crossings,fitResult,curve,selection,outputDir)
selectedIDs = selection.ID(selection.SelectedForFit);
use = ismember(crossings.ID,selectedIDs);
x = crossings.X_ab_RM(use);
rho = hypot(crossings.Y_ab_RM(use),crossings.Z_MSO_RM(use)-0.196);
theta = linspace(-2.55,2.55,1200);
ref = [0.5 1.04 2.86];
r = ref(3)./(1+ref(2)*cos(theta));
xRef = ref(1)+r.*cos(theta);
rhoRef = abs(r.*sin(theta));
valid = xRef>-6 & xRef<4 & rhoRef<8 & isfinite(xRef);

f = figure('Visible','off','Color','w','Position',[80 50 1200 950]);
theme(f,'light');
ax = axes(f); hold(ax,'on');
set(ax,'Color','w','XColor','k','YColor','k', ...
    'GridColor',[0.72 0.72 0.72]);
fill(ax,cos(linspace(0,2*pi,400)),sin(linspace(0,2*pi,400)), ...
    [0.72 0.72 0.72],'EdgeColor',[0.2 0.2 0.2], ...
    'DisplayName','Mercury (1 R_M)');
plot(ax,xRef(valid),rhoRef(valid),'Color',[0.92 0.65 0.05], ...
    'LineWidth',2.5,'DisplayName','Winslow et al. (2013)');
plot(ax,curve.X_ab_RM,curve.rho_RM,'r-','LineWidth',2.2, ...
    'DisplayName','Confirmed-point fit');
scatter(ax,x,rho,30,'k','filled','DisplayName','Confirmed crossings');
axis(ax,'equal'); xlim(ax,[-4 3]); ylim(ax,[0 5]); grid(ax,'on');
xlabel(ax,'X''_{MSO} (R_M)');
ylabel(ax,'\rho = [Y''^2+(Z-0.196)^2]^{1/2} (R_M)');
title(ax,sprintf(['Mercury bow-shock fit (confirmed points only)\n' ...
    'X_0=%.3f, \\epsilon=%.3f, L=%.3f, L_{SSP}=%.3f R_M'], ...
    fitResult.X0,fitResult.epsilon,fitResult.L,fitResult.L_SSP_RM));
lgd = legend(ax,'Location','northeast');
set(lgd,'Color','w','TextColor','k','EdgeColor',[0.35 0.35 0.35]);
exportgraphics(f,fullfile(outputDir,'page17_bowshock_fit.png'), ...
    'Resolution',300);
close(f);
end
