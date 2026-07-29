param(
    [switch]$SkipVisualReview
)

$runtimeRoot = Join-Path $env:USERPROFILE '.cache\codex-runtimes\codex-primary-runtime\dependencies'
$pythonExe = Join-Path $runtimeRoot 'python\python.exe'
$nodeExe = Join-Path $runtimeRoot 'node\bin\node.exe'
$nodeModules = Join-Path $runtimeRoot 'node\node_modules'
$nodeBin = Join-Path $runtimeRoot 'node\bin'
$overrideBin = Join-Path $runtimeRoot 'bin\override'
$fallbackBin = Join-Path $runtimeRoot 'bin\fallback'
$skillScripts = Join-Path $env:USERPROFILE '.codex\skills\presentation-skill\scripts'
$planningValidator = Join-Path $skillScripts 'validate_planning.py'
$qaGate = Join-Path $skillScripts 'qa_gate.py'
$visualReview = Join-Path $skillScripts 'visual_review.py'
$workspace = $PSScriptRoot
$outline = Join-Path $workspace 'outline.json'
$projectBuilder = Join-Path $workspace 'build_minimal_deck.js'
$builtDeck = Join-Path $workspace 'build\Spike_Topping_TypeIII_visual_simulation_optimized.pptx'
$qaDir = Join-Path $workspace 'build\qa'
$qaReport = Join-Path $qaDir 'report.json'
$finalDeck = [System.IO.Path]::GetFullPath((Join-Path $workspace '..\..\Spike_Topping_TypeIII_simulation.pptx'))

if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Bundled Python runtime not found: $pythonExe"
}
if (-not (Test-Path -LiteralPath $nodeExe)) {
    throw "Bundled Node.js runtime not found: $nodeExe"
}
foreach ($requiredPath in @($planningValidator, $qaGate, $visualReview, $outline, $projectBuilder)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Required rebuild input not found: $requiredPath"
    }
}

$env:NODE_PATH = $nodeModules
$env:PATH = "$nodeBin;$overrideBin;$fallbackBin;$env:PATH"

& $pythonExe $planningValidator --workspace $workspace
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

New-Item -ItemType Directory -Path (Split-Path -Parent $builtDeck) -Force | Out-Null
& $nodeExe $projectBuilder `
    --outline $outline `
    --output $builtDeck `
    --style-preset 'lab-report'
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
if (-not (Test-Path -LiteralPath $builtDeck)) {
    throw "Expected PPTX was not generated: $builtDeck"
}

New-Item -ItemType Directory -Path $qaDir -Force | Out-Null
& $pythonExe $qaGate `
    --input $builtDeck `
    --outdir $qaDir `
    --style-preset 'lab-report' `
    --strict-geometry `
    --skip-render `
    --skip-manual-review `
    --fail-on-visual-warnings `
    --fail-on-design-warnings `
    --outline $outline `
    --report $qaReport
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

if (-not $SkipVisualReview) {
    & $pythonExe $visualReview `
        --input $builtDeck `
        --outdir (Join-Path $qaDir 'visual_review') `
        --outline $outline `
        --skip-render `
        --fail-on-warnings
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}

Copy-Item -LiteralPath $builtDeck -Destination $finalDeck -Force
Write-Output "Final deck: $finalDeck"
