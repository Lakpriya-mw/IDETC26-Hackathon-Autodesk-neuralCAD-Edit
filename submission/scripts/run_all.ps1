# End-to-end run for lw_solution (Windows / PowerShell).
#
#   cd C:\0_IDETC_Hackathon\IDETC_Hackathon_Source
#   .\lw_solution\scripts\run_all.ps1
#
# Flags:
#   -Limit 3        run only 3 tasks (smoke test)
#   -SkipHarness    evaluate existing outputs without calling any model
#   -SkipEval       run the harness only

param(
    [int]$Limit = 0,
    [switch]$SkipHarness,
    [switch]$SkipEval,
    [switch]$SkipSelfTest
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$Config   = Join-Path $PSScriptRoot "..\config\lw_config.json" | Resolve-Path
$env:PYTHONPATH = $RepoRoot

Write-Host "repo root : $RepoRoot"
Write-Host "config    : $Config"

if (-not $SkipSelfTest) {
    Write-Host "`n=== 0. self-test ===" -ForegroundColor Cyan
    python "$PSScriptRoot\selftest.py" --config $Config
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Self-test failed. Fix the reported checks before running." -ForegroundColor Red
        exit 1
    }
}

if (-not $SkipHarness) {
    Write-Host "`n=== 1. run the agentic harness ===" -ForegroundColor Cyan
    $harnessArgs = @("--config", $Config)
    if ($Limit -gt 0) { $harnessArgs += @("--limit", $Limit) }
    python "$PSScriptRoot\run_harness.py" @harnessArgs
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    Write-Host "`n=== 2. backfill any missing .stl / views ===" -ForegroundColor Cyan
    python "$PSScriptRoot\postprocess.py"
}

if (-not $SkipEval) {
    Write-Host "`n=== 3. ingest + score with the organisers' pipeline ===" -ForegroundColor Cyan
    python "$PSScriptRoot\evaluate.py" --config $Config
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

Write-Host "`nDone. Open src/notebooks/leaderboard.ipynb to view the results." -ForegroundColor Green
