@echo off
REM ============================================================================
REM Video Transcriber - Launcher using Python 3.12
REM ============================================================================

echo.
echo ============================================================
echo    Video Transcriber - Starting...
echo ============================================================
echo.

REM Use Python 3.12 directly
set "PYTHON_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe"
set "PIP_CMD=C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe -m pip"

if not exist "%PYTHON_CMD%" (
    echo [ERROR] Python 3.12 not found at expected location
    pause
    exit /b 1
)

echo [OK] Python 3.12 found
"%PYTHON_CMD%" --version

REM Check FFmpeg
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    if exist "%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin\ffmpeg.exe" (
        set "PATH=%LOCALAPPDATA%\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin;%PATH%"
        echo [OK] FFmpeg found
    ) else (
        echo [ERROR] FFmpeg not found
        pause
        exit /b 1
    )
)

REM Check dependencies
echo.
echo [INFO] Checking dependencies...
"%PYTHON_CMD%" -c "import gradio" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Dependencies not installed. Installing now...
    echo This may take a few minutes. Please be patient.
    echo.

    REM Install with timeout and retry
    %PIP_CMD% install --default-timeout=1000 torch torchvision torchaudio
    if errorlevel 1 (
        echo [WARNING] PyTorch installation failed, retrying...
        %PIP_CMD% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    )

    %PIP_CMD% install --default-timeout=1000 gradio yt-dlp openai-whisper

    if errorlevel 1 (
        echo.
        echo [ERROR] Installation failed due to network timeout.
        echo.
        echo Please run this command manually to install dependencies:
        echo.
        echo   C:\Users\%USERNAME%\AppData\Local\Programs\Python\Python312\python.exe -m pip install torch torchvision torchaudio gradio yt-dlp openai-whisper
        echo.
        pause
        exit /b 1
    )

    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

echo.
echo [INFO] Starting application...
set "CUDA_VISIBLE_DEVICES=0"
"%PYTHON_CMD%" app.py

pause
