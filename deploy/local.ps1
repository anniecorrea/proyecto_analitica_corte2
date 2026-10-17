# Respaldo local del despliegue (Windows): levanta el frontend en este computador y, si se pide, lo publica con un
# Quick Tunnel de Cloudflare (https://*.trycloudflare.com), igual que el servidor de AWS. Sirve si el servidor falla
# el dia de la sustentacion. Usa la GPU si la hay (PENGWIN_DEVICE=auto).
#
# Desde la raiz del repositorio, en PowerShell:
#   powershell -ExecutionPolicy Bypass -File deploy\local.ps1                 # app + tunel de Cloudflare
#   powershell -ExecutionPolicy Bypass -File deploy\local.ps1 -SinTunel       # solo red local (sin limite de 100 MB)
#   powershell -ExecutionPolicy Bypass -File deploy\local.ps1 -Dispositivo cpu
# O doble clic en deploy\local.bat. Ctrl+C detiene la app y el tunel.
#
# Direcciones que imprime:
#   - este computador:           http://localhost:8501
#   - celular en la misma Wi-Fi: http://<IP-local>:8501   (Windows puede pedir permiso en el firewall la primera vez)
#   - cualquier red:             https://<algo>.trycloudflare.com  (el tunel no deja subir archivos de mas de 100 MB)
param(
    [switch]$SinTunel,
    [ValidateSet("auto", "cpu", "cuda")][string]$Dispositivo = "auto",
    [int]$Puerto = 8501
)
$ErrorActionPreference = "Stop"
$Raiz = Split-Path -Parent $PSScriptRoot
Set-Location $Raiz
$Registros = Join-Path $env:TEMP "pengwin_local"
# Se usa 127.0.0.1 y no localhost: en Windows localhost resuelve primero a ::1 (IPv6) y Streamlit solo escucha en IPv4
New-Item -ItemType Directory -Force $Registros | Out-Null

# ---------------------------------------------------------------- comprobaciones
$Streamlit = Join-Path $Raiz ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $Streamlit)) {
    throw "No encuentro .venv\Scripts\streamlit.exe. Crea el ambiente: python -m venv .venv; .venv\Scripts\python -m pip install -r requirements.txt"
}
foreach ($plano in "axial", "coronal", "sagital") {
    $modelo = Join-Path $Raiz "resultados\modelos\entrenamiento_multitarea_$plano\modelo_multitarea_$plano.pt"
    if (-not (Test-Path $modelo) -or (Get-Item $modelo).Length -lt 1MB) {
        throw "Falta $modelo o es un puntero de Git LFS. Ejecuta: git lfs install; git lfs pull"
    }
}
$enUso = Get-NetTCPConnection -LocalPort $Puerto -State Listen -ErrorAction SilentlyContinue
if ($enUso) { throw "El puerto $Puerto ya esta en uso (otra app de Streamlit abierta?). Cierrala o usa -Puerto 8502." }

$Cloudflared = $null
if (-not $SinTunel) {
    $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
    if ($cmd) { $Cloudflared = $cmd.Source }
    foreach ($ruta in "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe", "$env:ProgramFiles\cloudflared\cloudflared.exe") {
        if (-not $Cloudflared -and (Test-Path $ruta)) { $Cloudflared = $ruta }
    }
    if (-not $Cloudflared) {
        if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
            throw "Falta cloudflared. Instalalo desde https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/ o usa -SinTunel."
        }
        Write-Host "== Instalando cloudflared con winget (una sola vez)"
        winget install --id Cloudflare.cloudflared -e --accept-source-agreements --accept-package-agreements | Out-Host
        foreach ($ruta in "${env:ProgramFiles(x86)}\cloudflared\cloudflared.exe", "$env:ProgramFiles\cloudflared\cloudflared.exe") {
            if (-not $Cloudflared -and (Test-Path $ruta)) { $Cloudflared = $ruta }
        }
        if (-not $Cloudflared) { throw "winget termino pero no encuentro cloudflared.exe. Abre otra terminal y repite, o usa -SinTunel." }
    }
}

# ---------------------------------------------------------------- app
$env:PENGWIN_DEVICE = $Dispositivo
$app = $null
$tunel = $null
try {
    Write-Host "== Iniciando la app (PENGWIN_DEVICE=$Dispositivo, puerto $Puerto)"
    $app = Start-Process -FilePath $Streamlit -PassThru -WindowStyle Hidden `
        -ArgumentList "run", "app/streamlit_app.py", "--server.address", "0.0.0.0", "--server.port", "$Puerto", "--server.headless", "true" `
        -RedirectStandardOutput "$Registros\app.log" -RedirectStandardError "$Registros\app_error.log"
    $lista = $false
    for ($i = 0; $i -lt 90 -and -not $lista; $i++) {
        Start-Sleep -Seconds 1
        if ($app.HasExited) { throw "La app se cerro al arrancar. Revisa $Registros\app_error.log" }
        try { $lista = (Invoke-WebRequest -UseBasicParsing -TimeoutSec 2 "http://127.0.0.1:$Puerto/_stcore/health").Content -match "ok" } catch { }
    }
    if (-not $lista) { throw "La app no respondio en 90 s. Revisa $Registros\app_error.log" }

    $ip = Get-NetIPConfiguration | Where-Object { $_.IPv4DefaultGateway -and $_.NetAdapter.Status -eq "Up" } |
        Select-Object -First 1 -ExpandProperty IPv4Address | Select-Object -First 1 -ExpandProperty IPAddress

    # ------------------------------------------------------------ tunel
    $url = $null
    if ($Cloudflared) {
        Write-Host "== Abriendo el tunel de Cloudflare"
        $tunel = Start-Process -FilePath $Cloudflared -PassThru -WindowStyle Hidden `
            -ArgumentList "tunnel", "--no-autoupdate", "--url", "http://127.0.0.1:$Puerto" `
            -RedirectStandardOutput "$Registros\tunel.log" -RedirectStandardError "$Registros\tunel_error.log"
        for ($i = 0; $i -lt 60 -and -not $url; $i++) {
            Start-Sleep -Seconds 1
            if ($tunel.HasExited) { break }
            $m = Select-String -Path "$Registros\tunel_error.log", "$Registros\tunel.log" -Pattern "https://[a-z0-9-]+\.trycloudflare\.com" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($m) { $url = $m.Matches[0].Value }
        }
    }

    Write-Host ""
    Write-Host "Listo. La primera prediccion tarda un poco mas (carga los tres modelos)."
    Write-Host "  Este computador:           http://localhost:$Puerto"
    if ($ip) { Write-Host "  Celular en la misma Wi-Fi: http://${ip}:$Puerto" }
    if ($Cloudflared) {
        if ($url) { Write-Host "  Cualquier red (tunel):     $url   (archivos de hasta 100 MB)" }
        else { Write-Host "  El tunel no dio direccion. Revisa $Registros\tunel_error.log (la app sigue en la red local)." }
    }
    Write-Host "Registros: $Registros | Ctrl+C para detener."

    while (-not $app.HasExited) { Start-Sleep -Seconds 2 }
    Write-Host "La app se cerro. Revisa $Registros\app_error.log"
}
finally {
    foreach ($p in $tunel, $app) {
        if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue }
    }
    # streamlit.exe lanza python.exe como hijo: se detiene tambien
    Get-CimInstance Win32_Process -Filter "Name = 'python.exe' OR Name = 'streamlit.exe'" -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "streamlit" -and $_.CommandLine -match "streamlit_app.py" -and $_.CommandLine -match "$Puerto" } |
        ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
    Write-Host "App y tunel detenidos."
}
