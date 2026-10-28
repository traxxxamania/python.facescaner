@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title TIER CHECK

set "PORT=8000"
if not "%~1"=="" set "PORT=%~1"

echo.
echo  ============================================
echo    TIER CHECK - meme face tier scanner
echo    FastAPI + OpenCV YuNet
echo  ============================================
echo.

rem --- find a real Python launcher (avoid the Microsoft Store alias) -------
set "PY_CMD="
py --version >nul 2>nul
if not errorlevel 1 set "PY_CMD=py"
if not defined PY_CMD (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)
if not defined PY_CMD (
    echo [ERROR] Python 3.11+ was not found.
    echo         Install it from https://www.python.org/downloads/
    echo         and enable "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

rem --- virtual environment ------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Creating virtual environment...
    %PY_CMD% -m venv .venv
    if errorlevel 1 goto :fail
) else (
    echo [1/3] Virtual environment found.
)

rem --- dependencies -------------------------------------------------------
".venv\Scripts\python.exe" -c "import fastapi, uvicorn, cv2, numpy, PIL" >nul 2>nul
if errorlevel 1 (
    echo [2/3] Installing dependencies, this may take a minute...
    ".venv\Scripts\python.exe" -m pip install --upgrade pip >nul 2>nul
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt
    if errorlevel 1 goto :fail
) else (
    echo [2/3] Dependencies are ready.
)

rem --- make sure the detector model is present ----------------------------
if not exist "app\models\face_detection_yunet_2026may.onnx" (
    echo [ERROR] Face detector model is missing:
    echo         app\models\face_detection_yunet_2026may.onnx
    echo         Re-download the repository or see README.md.
    echo.
    pause
    exit /b 1
)

rem --- open the browser a couple of seconds after the server starts --------
echo [3/3] Starting server on http://127.0.0.1:%PORT%
if /i "%~2"=="nobrowser" goto :run
start "" powershell -NoProfile -WindowStyle Hidden -Command "Start-Sleep -Seconds 3; Start-Process 'http://127.0.0.1:%PORT%'"

:run
echo.
echo  Camera works on localhost without HTTPS.
echo  Press Ctrl+C in this window to stop the server.
echo.
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port %PORT%
if errorlevel 1 goto :fail
goto :eof

:fail
echo.
echo [ERROR] Something went wrong. Read the message above.
echo         If the port is busy, run:  start.bat 8001
echo.
pause
exit /b 1
