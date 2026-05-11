@echo off
setlocal

set VENV_DIR=.venv

rem Create venv if missing
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV_DIR%...
  python -m venv "%VENV_DIR%"
)

rem Install dependencies (idempotent)
echo Installing/updating dependencies...
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
call "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt

rem Run Agent 1
echo Running Agent 1...
call "%VENV_DIR%\Scripts\python.exe" main.py %*

endlocal

