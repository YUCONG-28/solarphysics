function [excluded, intervals] = mercury_bowshock_quality_mask(data)
%MERCURY_BOWSHOCK_QUALITY_MASK Identify periodic instrumental field bursts.
%   [MASK, INTERVALS] = MERCURY_BOWSHOCK_QUALITY_MASK(DATA) flags samples
%   within 120 seconds of a |B| > 1000 nT trigger. The padding removes the
%   complete calibration-like waveform rather than only clipping its peak.
%   The raw DATA array is never changed.

assert(isnumeric(data) && size(data,2)==7, ...
    'MercuryBowShock:QualitySchema', ...
    'Input must be numeric [datenum,X,Y,Z,Bx,By,Bz].');

time = data(:,1);
Bmag = vecnorm(data(:,5:7),2,2);
trigger = Bmag > 1000;
excluded = false(size(trigger));
paddingDays = 120/86400;

triggerIndex = find(trigger);
if isempty(triggerIndex)
    intervals = emptyIntervalTable();
    return
end

breaks = [1; find(diff(triggerIndex)>1)+1; numel(triggerIndex)+1];
for k = 1:numel(breaks)-1
    group = triggerIndex(breaks(k):breaks(k+1)-1);
    excluded = excluded | ...
        (time >= time(group(1))-paddingDays & ...
         time <= time(group(end))+paddingDays);
end

% The two bursts on each affected day overlap after padding. Rebuild the
% final merged intervals from the union mask for a concise audit trail.
maskIndex = find(excluded);
maskBreaks = [1; find(diff(maskIndex)>1)+1; numel(maskIndex)+1];
intervals = emptyIntervalTable();
for k = 1:numel(maskBreaks)-1
    group = maskIndex(maskBreaks(k):maskBreaks(k+1)-1);
    nExtreme = nnz(trigger(group));
    row = table( ...
        datetime(time(group(1)),'ConvertFrom','datenum','TimeZone','UTC'), ...
        datetime(time(group(end)),'ConvertFrom','datenum','TimeZone','UTC'), ...
        numel(group),nExtreme,max(Bmag(group)), ...
        "Periodic calibration-like |B| burst; 120 s padding", ...
        'VariableNames',intervals.Properties.VariableNames);
    intervals = [intervals; row]; %#ok<AGROW>
end
end

function t = emptyIntervalTable()
t = table('Size',[0 6], ...
    'VariableTypes',{'datetime','datetime','double','double','double','string'}, ...
    'VariableNames',{'StartUTC','EndUTC','ExcludedSamples','ExtremeSamples', ...
    'MaxB_nT','Reason'});
t.StartUTC.TimeZone = 'UTC';
t.EndUTC.TimeZone = 'UTC';
end
