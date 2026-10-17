@echo off
rem Doble clic: respaldo local del despliegue (app + tunel de Cloudflare). Ver deploy\local.ps1.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0local.ps1" %*
pause
