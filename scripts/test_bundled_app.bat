@echo off
REM Test script to build and run the bundled Windows app locally.
REM
REM PyInstaller produces a onedir build: dist\Jarvis\Jarvis.exe next to an
REM _internal folder that holds the data files the app reads at runtime.
REM scripts\check_bundle_layout.py derives those files from the source tree and
REM names any that the build is missing, before the app is launched.

echo.
echo === Building Jarvis Desktop App with PyInstaller ===
echo.

REM Get to project root
cd /d "%~dp0\.."

REM Set up paths
set "PROJECT_ROOT=%cd%"
set "MAMBA_ENV=%PROJECT_ROOT%\.mamba_env"
set "PYTHONPATH=%PROJECT_ROOT%\src;%PYTHONPATH%"

REM Check if mamba environment exists
if not exist "%MAMBA_ENV%\python.exe" (
    echo ERROR: Mamba environment not found at %MAMBA_ENV%
    echo    Please run the setup script first.
    pause
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist "build" rmdir /s /q build
if exist "dist" rmdir /s /q dist
echo.

REM Build with PyInstaller
echo Building app bundle...
"%MAMBA_ENV%\python.exe" -m PyInstaller jarvis_desktop.spec
echo.

REM Check the executable the Windows build produces
if not exist "dist\Jarvis\Jarvis.exe" (
    echo Build failed! dist\Jarvis\Jarvis.exe was not produced. Check the output above for errors.
    exit /b 1
)

REM Check the bundle carries every data file and licence text where it belongs
echo Checking the build layout...
"%MAMBA_ENV%\python.exe" scripts\check_bundle_layout.py --dist dist --plain
if errorlevel 1 (
    echo.
    echo Build is incomplete: the files listed above are missing from dist\Jarvis\_internal.
    echo Add them to the datas list in jarvis_desktop.spec.
    exit /b 1
)
echo.

echo Build successful!
echo.
echo App location: %cd%\dist\Jarvis\Jarvis.exe
echo.

REM Show file info
echo File info:
dir dist\Jarvis\Jarvis.exe
echo.

REM Run the app
echo Launching app...
echo    Press Ctrl+C in this window to stop the app
echo.

dist\Jarvis\Jarvis.exe

echo.
echo App exited.
