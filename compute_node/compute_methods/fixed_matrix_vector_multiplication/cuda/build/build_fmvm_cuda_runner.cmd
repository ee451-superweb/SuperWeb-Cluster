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
nvcc ..\fmvm_cuda_runner.cu -O3 --use_fast_math -std=c++17 -cudart static -Xcompiler "/MT /EHsc" -o fmvm_cuda_runner.exe -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_120,code=sm_120
set "BUILD_EXIT=%ERRORLEVEL%"
popd
exit /b %BUILD_EXIT%
