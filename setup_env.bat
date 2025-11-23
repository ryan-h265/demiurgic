@echo off
REM Setup script for Demiurgic virtual environment (Windows)
REM
REM Usage:
REM   setup_env.bat           Quick setup (core only)
REM   setup_env.bat --full    Full setup (with training deps)

echo ===============================================================
echo          Demiurgic Virtual Environment Setup (Windows)
echo ===============================================================
echo.

REM Check Python version
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please install Python 3.10+ and add it to PATH
    pause
    exit /b 1
)

echo Found Python:
python --version
echo.

REM Create virtual environment
set VENV_DIR=venv

if exist %VENV_DIR% (
    echo Warning: Virtual environment already exists at .\%VENV_DIR%
    set /p RECREATE="Do you want to recreate it? (y/N): "
    if /i "%RECREATE%"=="y" (
        echo Removing existing environment...
        rmdir /s /q %VENV_DIR%
    ) else (
        echo Skipping environment creation
        echo.
        echo To activate existing environment, run:
        echo   %VENV_DIR%\Scripts\activate
        pause
        exit /b 0
    )
)

echo Creating virtual environment...
python -m venv %VENV_DIR%
echo Virtual environment created at .\%VENV_DIR%
echo.

REM Activate virtual environment
call %VENV_DIR%\Scripts\activate.bat
echo Virtual environment activated
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip setuptools wheel
echo pip upgraded
echo.

REM Install dependencies
if "%1"=="--full" (
    echo Installing FULL dependencies (this may take a while)...
    echo.
    pip install -r requirements.txt
    echo.
    echo All dependencies installed
) else (
    echo Installing CORE dependencies (minimal setup)...
    echo.
    pip install -r requirements-core.txt
    echo.
    echo Core dependencies installed
    echo.
    echo Note: To install training dependencies later, run:
    echo   venv\Scripts\activate
    echo   pip install -r requirements-training.txt
)

echo.
echo ===============================================================
echo                    Setup Complete!
echo ===============================================================
echo.
echo To activate the environment:
echo   venv\Scripts\activate
echo.
echo To test the model:
echo   venv\Scripts\activate
echo   python scripts\test_model_basic.py
echo.
echo To deactivate when done:
echo   deactivate
echo.
pause
