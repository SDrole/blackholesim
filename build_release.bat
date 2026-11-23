@echo off
setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo ERROR: Visual Studio Installer / vswhere.exe was not found.
  exit /b 1
)

for /f "usebackq tokens=*" %%I in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do (
  set "VSINSTALL=%%I"
)

if not defined VSINSTALL (
  echo ERROR: Visual Studio C++ Build Tools were not found.
  exit /b 1
)

call "%VSINSTALL%\VC\Auxiliary\Build\vcvars64.bat"
if errorlevel 1 exit /b 1

where cmake >nul 2>nul
if errorlevel 1 (
  echo ERROR: cmake was not found on PATH.
  exit /b 1
)

where ninja >nul 2>nul
if errorlevel 1 (
  echo ERROR: ninja was not found on PATH.
  exit /b 1
)

cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
if errorlevel 1 exit /b 1

cmake --build build
