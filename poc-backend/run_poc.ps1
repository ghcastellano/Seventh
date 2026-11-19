# Um clique para subir Mock Situator (8090) + API POC (8000) e rodar o teste E2E
$ErrorActionPreference = "Stop"
$dir = Split-Path -Parent $MyInvocation.MyCommand.Path
$py = Join-Path $dir ".venv\Scripts\python.exe"

if (!(Test-Path $py)) {
  Write-Host "Ambiente .venv nao encontrado. Execute: uv venv .venv --python 3.11 && uv pip install -r requirements.txt --python .venv\Scripts\python.exe" -ForegroundColor Red
  exit 1
}

$env:SITUATOR_WEBHOOK_URL = "http://127.0.0.1:8090/webhook"

Write-Host "[1/3] Subindo Mock Situator Webhook na porta 8090..." -ForegroundColor Cyan
$mockProc = Start-Process -FilePath $py -ArgumentList "-m uvicorn mock_situator_webhook:app --host 127.0.0.1 --port 8090" -WorkingDirectory $dir -PassThru -WindowStyle Minimized

Write-Host "[2/3] Subindo API POC (YOLOv8-Pose + VLM) na porta 8000..." -ForegroundColor Cyan
$apiProc = Start-Process -FilePath $py -ArgumentList "-m uvicorn main:app --host 127.0.0.1 --port 8000" -WorkingDirectory $dir -PassThru -WindowStyle Minimized

Start-Sleep -Seconds 4
Write-Host "[3/3] Executando teste E2E (test_e2e.py)..." -ForegroundColor Green
& $py (Join-Path $dir "test_e2e.py")

Write-Host "`nServidores ativos! Acesse http://127.0.0.1:8000/docs ou use http://127.0.0.1:8000/analyze na aba Demo POC do site." -ForegroundColor Yellow
Write-Host "PIDs -> Mock Situator: $($mockProc.Id) | API POC: $($apiProc.Id)" -ForegroundColor DarkGray
