@echo off
setlocal
set "VSDEVCMD="
if exist "%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" (
  for /f "usebackq delims=" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -find Common7\Tools\VsDevCmd.bat`) do set "VSDEVCMD=%%i"
)
if not defined VSDEVCMD set "VSDEVCMD=%ProgramFiles(x86)%\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat"
if not exist "%VSDEVCMD%" exit /b 1
call "%VSDEVCMD%" -arch=x64 -host_arch=x64 >nul
pushd "%~dp0"
cl /nologo /std:c++20 /O2 /fp:fast /MT /EHsc /Fe:gemv_cpu_windows.exe ..\gemv_cpu_windows.cpp
set "BUILD_EXIT=%ERRORLEVEL%"
popd
exit /b %BUILD_EXIT%
