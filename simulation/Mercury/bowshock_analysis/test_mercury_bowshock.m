function test_mercury_bowshock
%TEST_MERCURY_BOWSHOCK Regression checks for quality masking and fitting.

analysisDir = fileparts(mfilename('fullpath'));
dataDir = fullfile(fileparts(analysisDir),'201312_01s');
saved = load(fullfile(analysisDir,'bowshock_crossings.mat'));

assert(height(saved.exclusions)==4, ...
    'Expected four merged periodic-artifact intervals.');
assert(sum(saved.exclusions.ExtremeSamples)==160, ...
    'Expected 160 |B| > 1000 nT trigger samples.');
assert(sum(saved.exclusions.ExcludedSamples)==1283, ...
    'Expected 1,283 samples after 120 s padding.');
assert(saved.validation.CandidateExcludedCount==0, ...
    'No automatic crossing should overlap an artifact interval.');

loaded = load(fullfile(dataDir,'20131204_01s.mat'),'data_mso');
day = datenum(datetime(2013,12,4,'TimeZone','UTC'));
d = loaded.data_mso(loaded.data_mso(:,1)>=day & ...
    loaded.data_mso(:,1)<day+1,:);
[mask,intervals] = mercury_bowshock_quality_mask(d);
assert(height(intervals)==1 && nnz(mask)==321, ...
    'December 4 artifact mask is not reproducible.');
assert(all(vecnorm(d(~mask,5:7),2,2)<=1000), ...
    'An extreme trigger escaped the quality mask.');

% Add a deliberately inner duplicate five minutes after the first inbound
% point. The final fit must keep only the outer point on that orbital leg.
sample = saved.crossings(1:4,:);
sample.Status(:) = "Confirmed";
duplicate = sample(1,:);
duplicate.ID = max(sample.ID)+1;
duplicate.UTC = duplicate.UTC + minutes(5);
duplicate.X_ab_RM = 0.8*duplicate.X_ab_RM;
duplicate.Y_ab_RM = 0.8*duplicate.Y_ab_RM;
duplicate.Z_MSO_RM = 0.8*duplicate.Z_MSO_RM;
duplicate.X_MSO_km = 0.8*duplicate.X_MSO_km;
duplicate.Y_MSO_km = 0.8*duplicate.Y_MSO_km;
duplicate.Z_MSO_km = 0.8*duplicate.Z_MSO_km;
sample = [sample; duplicate];
[fitResult,~,selection] = mercury_bowshock_fit(sample,false);
assert(fitResult.NInputPoints==5 && fitResult.NPoints==4 && ...
    fitResult.NDiscardedInner==1, ...
    'The inner duplicate was not discarded from the final fit.');
assert(nnz(selection.SelectedForFit)==4, ...
    'Fit selection audit has an unexpected row count.');

fprintf(['Mercury bow-shock regression checks passed: 4 artifact intervals, ' ...
    '1,283 excluded samples, and outermost-only fitting verified.\n']);
end
