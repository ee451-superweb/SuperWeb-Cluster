@echo off
REM Double-click entry point: prepare the project environment, then hand off to
REM bootstrap.py with UAC elevation and exit. Setup is idempotent, so repeated
REM double-clicks only pay the real cost on the first run or after a
REM requirements change. Bootstrap self-elevates via ShellExecute "runas",
REM which by Windows design spawns its own administrator console; this script
REM does not try to hold that console open.

setlocal
pushd "%~dp0"
chcp 65001 >NUL

set "HOST_PY="
where py >NUL 2>&1 && set "HOST_PY=py -3"
if not defined HOST_PY (
    where python >NUL 2>&1 && set "HOST_PY=python"
)
if not defined HOST_PY (
    echo [ERROR] No Python interpreter found on PATH.
    echo Install Python 3.10+ from https://www.python.org/downloads/ and rerun this launcher.
    pause
    popd
    endlocal
    exit /b 1
)

echo [run.cmd] Running setup with %HOST_PY%...
%HOST_PY% -X utf8 setup.py
if errorlevel 1 (
    echo [ERROR] setup.py exited with code %errorlevel%. Fix the errors above and rerun this launcher.
    pause
    popd
    endlocal
    exit /b %errorlevel%
)

set "VENV_PY=.venv\Scripts\python.exe"
if not exist "%VENV_PY%" (
    echo [ERROR] Expected %VENV_PY% to exist after setup, but it is missing.
    pause
    popd
    endlocal
    exit /b 1
)

echo [run.cmd] Launching bootstrap. Accept the UAC prompt when it appears; bootstrap will open its own administrator console.
start "" "%VENV_PY%" -X utf8 bootstrap.py --elevate-if-needed

popd
endlocal
exit /b 0
