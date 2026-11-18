@echo off
REM Laptop 1 only (Laptop 2 needs just a browser).
cd /d "%~dp0.."
py -m pip install -r requirements.txt
pause
