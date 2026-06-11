@echo off
REM TensorFlow requires Python 3.12 (NOT 3.14)
cd /d "%~dp0"

echo Creating virtual environment...
py -3.12 -m venv .venv

echo Installing packages (5-10 min, ~400 MB download)...
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\pip install -r requirements.txt

echo.
.venv\Scripts\python -c "import tensorflow as tf; print('TensorFlow OK:', tf.__version__)"

echo.
echo ============================================
echo  USE THESE COMMANDS (not plain python):
echo ============================================
echo   .venv\Scripts\activate
echo   python train_model.py
echo   python app.py
echo.
pause
