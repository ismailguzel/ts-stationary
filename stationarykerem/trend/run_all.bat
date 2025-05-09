@echo off
call venv\Scripts\activate
python data\generate_dataset.py
python models\train_lstm.py
python models\train_xgboost.py
pause
