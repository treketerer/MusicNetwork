@echo off
echo [1/2] Updating pip inside virtual environment...
".venv\Scripts\python.exe" -m pip install --upgrade pip

echo [2/2] Installing dependencies from requirements.txt...
".venv\Scripts\python.exe" -m pip install -r "requirements.txt"

echo.
echo Done! All libraries are updated.
pause