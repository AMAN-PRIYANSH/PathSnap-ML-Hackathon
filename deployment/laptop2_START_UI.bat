@echo off
REM ===== LAPTOP 2: opens the evaluator UI in the browser. No Python needed, works offline. =====
REM In the page, type Laptop 1's address, e.g. http://192.168.1.23:8000, then click "Load trips".
start "" "%~dp0..\ui\index.html"
