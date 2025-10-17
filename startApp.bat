@echo off
REM ========================================
REM Startup Script for Image Analysis Application
REM ========================================
setlocal EnableDelayedExpansion

REM Wait for system to fully boot
timeout /t 30 /nobreak > nul

REM Define paths
set "PROJECT_DIR=D:\AVANATICA Q\Bin"
set "PYTHON_EXE=D:\AVANATICA Q\venv\python-3.10.11-embed-amd64\python.exe"
set "MAIN_SCRIPT=startModel_Updated.py"
set "LOG_FILE=%PROJECT_DIR%\startup.log"

REM Create log file with timestamp
echo %date% %time% - Starting Image Analysis Application >> "%LOG_FILE%"

:MAIN_LOOP
echo ========================================
echo Image Analysis Application Startup
echo ========================================
echo Project Directory: %PROJECT_DIR%
echo Python Executable: %PYTHON_EXE%
echo Main Script: %MAIN_SCRIPT%
echo ========================================

REM Change to project directory
cd /d "%PROJECT_DIR%"
if errorlevel 1 (
    echo ERROR: Failed to change to project directory!
    echo %date% %time% - ERROR: Failed to change to project directory >> "%LOG_FILE%"
    pause
    exit /b 1
)

REM Check if Python executable exists
if not exist "%PYTHON_EXE%" (
    echo ERROR: Python executable not found at %PYTHON_EXE%
    echo %date% %time% - ERROR: Python executable not found >> "%LOG_FILE%"
    pause
    exit /b 1
)

REM Check if main script exists
if not exist "%MAIN_SCRIPT%" (
    echo ERROR: Main script %MAIN_SCRIPT% not found in project directory!
    echo %date% %time% - ERROR: Main script not found >> "%LOG_FILE%"
    pause
    exit /b 1
)

echo Python version:
"%PYTHON_EXE%" --version

echo ========================================
echo Starting application...
echo %date% %time% - Application started >> "%LOG_FILE%"
echo ========================================

REM Run the Python application
"%PYTHON_EXE%" "%MAIN_SCRIPT%"

REM Check exit code
set "EXIT_CODE=%errorlevel%"
echo.
echo ========================================
echo Application exited with code: %EXIT_CODE%
echo %date% %time% - Application exited with code %EXIT_CODE% >> "%LOG_FILE%"

if %EXIT_CODE% equ 0 (
    echo Application completed successfully.
    echo %date% %time% - Application completed successfully >> "%LOG_FILE%"
) else (
    echo Application crashed or exited with error!
    echo %date% %time% - Application crashed with error code %EXIT_CODE% >> "%LOG_FILE%"
)

REM Auto-restart logic
echo ========================================
echo Auto-restart in 10 seconds...
echo ========================================
timeout /t 10 /nobreak > nul

if %EXIT_CODE% neq 0 (
    echo %date% %time% - Auto-restarting application >> "%LOG_FILE%"
    echo Restarting application...
    goto MAIN_LOOP
) else (
    echo Normal termination - not restarting
    echo %date% %time% - Normal termination >> "%LOG_FILE%"
)

echo ========================================
echo Script completed
echo ========================================
pause