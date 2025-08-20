@echo off
REM Setup script to create data directory structure for each experiment

echo Setting up data directory structure...

REM Create data directories for each experiment
mkdir batch\cyber 2>nul
mkdir online\cyber 2>nul
mkdir cl_case1\cyber 2>nul
mkdir cl_case2\cyber 2>nul

echo Created directories:
echo   - batch\cyber\      (batch learning data)
echo   - online\cyber\     (online learning data)
echo   - cl_case1\cyber\   (continual learning case 1 data)
echo   - cl_case2\cyber\   (continual learning case 2 data)
echo.
echo ✅ Setup complete! Each experiment has its own data directory:
echo    - No conflicts between experiments
echo    - Automatic data download on first run
echo    - Independent caching for faster subsequent runs
echo.
echo You can now run any experiment - they'll handle data automatically!
