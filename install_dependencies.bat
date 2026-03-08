@echo off
REM ============================================================================
REM Video Transcriber - Dependencies Installer
REM ============================================================================

echo.
echo ============================================================
echo    Video Transcriber - Installing Dependencies
echo ============================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not found in PATH
    echo Please install Python 3.10+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/3] Upgrading pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Installing PyTorch with CUDA support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo.
echo [3/3] Installing application dependencies...
pip install gradio yt-dlp openai-whisper

echo.
echo ============================================================
echo    Installation Complete!
echo ============================================================
echo.
echo You can now run run.bat to start the application
echo.

pause
