@echo off
REM ===== LAPTOP 2: checks that Laptop 1's API can be reached over the network. =====
set /p IP=Type Laptop 1's IP address (example 192.168.1.23) and press Enter: 
echo.
echo --- /health ---
curl -s http://%IP%:8000/health
echo.
echo --- /trips ---
curl -s http://%IP%:8000/trips
echo.
echo.
echo If you see JSON above, the connection works. If not: same Wi-Fi? Firewall "Allow" on Laptop 1? API window open?
pause
