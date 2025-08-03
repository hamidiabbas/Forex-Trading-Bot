echo Setting up Forex Trading Bot...

echo Installing Python packages...
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3[extra]
pip install transformers>=4.21.0
pip install gymnasium
pip install yfinance pandas numpy matplotlib seaborn scikit-learn

echo Creating directories...
mkdir logs\production_training 2>nul
mkdir models 2>nul
mkdir results 2>nul

echo Setup complete!
echo Run: python enhanced_train_rl_model.py
pause