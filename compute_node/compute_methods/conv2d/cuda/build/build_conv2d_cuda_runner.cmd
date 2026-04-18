@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\Tools\VsDevCmd.bat" -arch=x64 -host_arch=x64 >nul
if errorlevel 1 exit /b %errorlevel%
pushd "%~dp0"
nvcc -std=c++17 -O3 --use_fast_math -Wno-deprecated-gpu-targets -cudart static -Xcompiler "/MT /EHsc" -o conv2d_cuda_runner.exe ..\conv2d_cuda_runner.cu -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_87,code=sm_87 -gencode=arch=compute_88,code=sm_88 -gencode=arch=compute_89,code=sm_89 -gencode=arch=compute_90,code=sm_90 -gencode=arch=compute_100,code=sm_100 -gencode=arch=compute_103,code=sm_103 -gencode=arch=compute_110,code=sm_110 -gencode=arch=compute_120,code=sm_120 -gencode=arch=compute_121,code=sm_121 -gencode=arch=compute_120,code=compute_120
set "BUILD_EXIT=%ERRORLEVEL%"
popd
exit /b %BUILD_EXIT%
