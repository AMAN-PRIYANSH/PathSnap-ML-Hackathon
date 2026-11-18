@echo off
REM ===== LAPTOP 1: model + API. Keep this window open: it shows the request logs. =====
cd /d "%~dp0.."
REM Which partition the UI may choose from: evaluator_test (final evaluation) or student_test (practice).
set "PATHSNAP_PARTITION=evaluator_test"
REM Stop an older API that may still be using port 8000.
for /f "tokens=5" %%p in ('netstat -ano ^| findstr ":8000 " ^| findstr LISTENING') do (
  echo Stopping an older program that is using port 8000 - process %%p
  taskkill /PID %%p /F >nul
)
echo.
echo This laptop's IP address (type it into the UI on Laptop 2):
ipconfig | findstr /i "IPv4"
echo.
echo If Windows Firewall asks, click "Allow" for Python on private networks.
echo.
py -m uvicorn api.app:app --host 0.0.0.0 --port 8000
pause
