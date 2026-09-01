#!/bin/sh
# Each stage is resumable, so re-running this script after any failure picks up
# where it stopped rather than starting over.
set -e
cd "C:/Users/shusm/Downloads/razorpya/fraud-risk-manager"
python -u -m src.tune   --trials 30 2>&1 | tee -a reports/tune_log.txt
python -u -m src.train              2>&1 | tee -a reports/train_log.txt
python -u -m src.assess             2>&1 | tee -a reports/assess_log.txt
python -u -m src.context            2>&1 | tee -a reports/context_log.txt
echo "CHAIN COMPLETE"
