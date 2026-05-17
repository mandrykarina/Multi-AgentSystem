@echo off
setlocal

echo Running full article experiment (100 tasks x 3 seeds)...
call "..\.venv\Scripts\python.exe" "..\main.py" --dataset --all --runs 3 --seeds 42,43,44
if errorlevel 1 (
  echo Full experiment failed.
  exit /b 1
)

echo Validating final summary contract...
call "..\.venv\Scripts\python.exe" "validate_final_summary.py"
if errorlevel 1 (
  echo Final summary validation failed.
  exit /b 1
)

echo Checking dataset distribution...
call "..\.venv\Scripts\python.exe" "check_dataset_distribution.py"
if errorlevel 1 (
  echo Dataset distribution check failed.
  exit /b 1
)

echo Full experiment completed and validated.
endlocal
