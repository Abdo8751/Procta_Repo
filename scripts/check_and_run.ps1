# Run after every change: tests first, then the dashboard (if not running) and the live replay
# with the current code.
#
#   .\scripts\check_and_run.ps1                          # Test_1 replay, context mode, YOLO-World + CLIP
#   .\scripts\check_and_run.ps1 --mode improved          # any extra live.py options are passed on
#   .\scripts\check_and_run.ps1 -NoTest                  # skip the tests
#   .\scripts\check_and_run.ps1 -Video data/videos/other.mp4
param([string]$Video = "data/videos/Test_1.mp4", [switch]$NoTest,
      [Parameter(ValueFromRemainingArguments = $true)] [string[]]$LiveArgs)

Set-Location (Join-Path $PSScriptRoot "..")
$py = ".venv\Scripts\python.exe"

if (-not $NoTest) {
    Write-Host "== tests"
    & $py -m pytest -q
    if ($LASTEXITCODE -ne 0) { Write-Host "Tests failed - fix them before running the demo."; exit 1 }
}

$running = $false
try { Invoke-WebRequest -Uri "http://127.0.0.1:8765/" -UseBasicParsing -TimeoutSec 2 | Out-Null; $running = $true } catch {}
if (-not $running) {
    Write-Host "== starting dashboard -> http://localhost:8765"
    Start-Process -FilePath $py -ArgumentList "scripts/dashboard.py" -WindowStyle Minimized
    Start-Sleep -Seconds 2
} else {
    Write-Host "== dashboard already running (restart it yourself if you changed dashboard.py)"
}

Write-Host "== live replay of $Video  (space pause, j/l -/+10 s, +/- speed, k pose, q quit)"
& $py scripts/live.py --mode context --video $Video --start-point T1 `
    --object-detector world --imgsz 640 --det-imgsz 1280 --det-skip 2 --stride 2 @LiveArgs
