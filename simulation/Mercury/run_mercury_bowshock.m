function run_mercury_bowshock
%RUN_MERCURY_BOWSHOCK One-click portable launcher.
%   Open this file in MATLAB and press Run. The launcher uses paths relative
%   to itself, extracts the bundled data on first use, and opens the review
%   application. No source MAT file is modified.

packageDir = fileparts(mfilename('fullpath'));
analysisDir = fullfile(packageDir,'bowshock_analysis');
dataDir = fullfile(packageDir,'201312_01s');
dataZip = fullfile(packageDir,'201312_01s.zip');
outputDir = fullfile(packageDir,'results');

assert(isfolder(analysisDir), ...
    'MercuryBowShock:MissingAnalysis', ...
    'Missing analysis directory: %s',analysisDir);

if ~isfolder(dataDir)
    assert(isfile(dataZip), ...
        'MercuryBowShock:MissingDataArchive', ...
        'Missing data archive: %s',dataZip);
    fprintf('First run: extracting 31 daily MAT files...\n');
    unzip(dataZip,dataDir);
end

files = dir(fullfile(dataDir,'201312*_01s.mat'));
assert(numel(files)==31, ...
    'MercuryBowShock:PortableFileCount', ...
    'Expected 31 MAT files after extraction, found %d.',numel(files));

if ~isfolder(outputDir)
    mkdir(outputDir);
end

addpath(analysisDir);
cleanupPath = onCleanup(@() rmpath(analysisDir)); %#ok<NASGU>
mercury_bowshock_app(dataDir,outputDir);
end
