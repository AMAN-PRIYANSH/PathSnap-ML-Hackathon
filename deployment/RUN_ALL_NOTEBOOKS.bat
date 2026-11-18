@echo off
setlocal EnableDelayedExpansion
REM Re-runs all five step notebooks in order on THIS laptop and saves their outputs.
REM Log: logs\run_all.txt   (about 20-25 minutes)
cd /d "%~dp0.."
if not exist logs mkdir logs
echo PathSnap: running notebooks 01-05 on this laptop. Log: logs\run_all.txt  -- do not close this window.
echo === %date% %time% start === > logs\run_all.txt
for %%N in (01_EDA 02_Preprocessing_and_Feature_Engineering 03_Model_Training_and_Comparison 04_Bayesian_Optimisation 05_HMM_Viterbi) do (
  echo === !date! !time! %%N === >> logs\run_all.txt
  echo running %%N ...
  py -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=7200 notebooks\%%N.ipynb >> logs\run_all.txt 2>&1
)
echo === %date% %time% DONE === >> logs\run_all.txt
echo DONE. You can close this window.
pause
