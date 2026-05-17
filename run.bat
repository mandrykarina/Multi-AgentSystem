@echo off
setlocal

set VENV_DIR=.venv
set REQUIRED_PY=3.11
set PYTHON_BIN=

rem 1) Resolve Python 3.11 explicitly
for /f "delims=" %%i in ('py -3.11 -c "import sys; print(sys.executable)" 2^>nul') do (
  set "PYTHON_BIN=%%i"
)
if not defined PYTHON_BIN (
  if exist "%LocalAppData%\Programs\Python\Python311\python.exe" (
    set "PYTHON_BIN=%LocalAppData%\Programs\Python\Python311\python.exe"
  )
)
if not defined PYTHON_BIN (
  if exist "%ProgramFiles%\Python311\python.exe" (
    set "PYTHON_BIN=%ProgramFiles%\Python311\python.exe"
  )
)
if not defined PYTHON_BIN (
  if exist "%ProgramFiles(x86)%\Python311\python.exe" (
    set "PYTHON_BIN=%ProgramFiles(x86)%\Python311\python.exe"
  )
)
if not defined PYTHON_BIN (
  echo Python %REQUIRED_PY% was not found.
  echo Install Python 3.11 and run this script again.
  echo Download: https://www.python.org/downloads/release/python-3119/
  exit /b 1
)

rem 2) Ensure resolved interpreter is really 3.11
set RESOLVED_PY=
for /f "tokens=2 delims= " %%i in ('"%PYTHON_BIN%" -V 2^>^&1') do (
  set "RESOLVED_PY=%%i"
)
set "RESOLVED_PY=%RESOLVED_PY:~0,4%"
if /i not "%RESOLVED_PY%"=="%REQUIRED_PY%" (
  echo Resolved interpreter is %RESOLVED_PY%, but %REQUIRED_PY% is required.
  echo Interpreter path: %PYTHON_BIN%
  exit /b 1
)

rem 3) Create/recreate venv strictly with Python 3.11
if not exist "%VENV_DIR%\Scripts\python.exe" (
  echo Creating virtual environment in %VENV_DIR% with Python %REQUIRED_PY%...
  "%PYTHON_BIN%" -m venv "%VENV_DIR%"
) else (
  set CURRENT_VENV_PY=
  for /f "tokens=2 delims= " %%i in ('"%VENV_DIR%\Scripts\python.exe" -V 2^>^&1') do (
    set "CURRENT_VENV_PY=%%i"
  )
  set "CURRENT_VENV_PY=%CURRENT_VENV_PY:~0,4%"
  if /i not "%CURRENT_VENV_PY%"=="%REQUIRED_PY%" (
    echo Existing venv uses Python %CURRENT_VENV_PY%. Recreating with %REQUIRED_PY%...
    rmdir /s /q "%VENV_DIR%"
    "%PYTHON_BIN%" -m venv "%VENV_DIR%"
  )
)

rem Install dependencies (idempotent)
echo Installing/updating dependencies...
call "%VENV_DIR%\Scripts\python.exe" -m pip install --upgrade pip
call "%VENV_DIR%\Scripts\python.exe" -m pip install -r requirements.txt

if "%~1"=="" (
  rem Run reproducible experiment by default (no args)
  echo Running reproducible experiment...
  call "%VENV_DIR%\Scripts\python.exe" main.py --dataset --all --runs 3 --seeds 42,43,44
) else (
  rem Pass-through mode for custom CLI arguments
  echo Running custom command...
  call "%VENV_DIR%\Scripts\python.exe" main.py %*
)

endlocal

