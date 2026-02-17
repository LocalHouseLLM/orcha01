@echo off

echo Creating virtual environment...
python -m venv venv

echo Activating venv...
call venv\Scripts\activate

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing PyTorch with CUDA...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo Installing other dependencies...
pip install -r requirements.txt

echo.
echo ✅ Installation complete.
echo To activate later use:
echo venv\Scripts\activate
pause
