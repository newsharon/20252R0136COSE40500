# run_fetch.ps1  
$ProjectRoot = 'C:\Users\newsh\OneDrive\바탕화~1\쿠빅\CHARTD~1'  # Short (8.3) path

Set-Location -LiteralPath $ProjectRoot

if (!(Test-Path -LiteralPath ".\logs")) {
  New-Item -Type Directory -Path ".\logs" | Out-Null
}

$py = Join-Path $ProjectRoot ".\.venv\Scripts\python.exe"
if (!(Test-Path $py)) { $py = "python" }

& $py ".\fetch_data.py" *>> ".\logs\fetch.log"
