#!/usr/bin/env bash
# Usage: train_watchdog.sh NAME [train.py args for a fresh start...]
# Starts runs/NAME (or resumes it from last.pt) and keeps resuming after GPU out-of-memory
# crashes, waiting until other apps (games, Steam) free the GPU first.
cd "$(dirname "$0")/.." || exit 1
NAME=$1; shift
RUN=runs/$NAME
batch=""; fails=0; attempt=0  # empty: resume keeps the run's own batch; set to 8 after repeated OOMs

gpu_free_mb() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' \r'; }
train_procs() { powershell -NoProfile -Command "@(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train.py*' -or \$_.CommandLine -like '*multiprocessing.spawn*' }).Count" | tr -d ' \r'; }
kill_leftovers() { powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*train.py*' -or \$_.CommandLine -like '*multiprocessing.spawn*' } | ForEach-Object { taskkill /PID \$_.ProcessId /T /F 2>\$null | Out-Null }"; }

while :; do
  attempt=$((attempt + 1)); waited=0
  until [ "$(gpu_free_mb)" -ge 4800 ] && [ "$(train_procs)" -eq 0 ]; do
    [ $waited -eq 0 ] && echo "waiting for GPU: free=$(gpu_free_mb) MB, leftover train processes=$(train_procs) ($(date +%H:%M))"
    waited=1; sleep 30
  done
  LOG=$RUN.$(date +%m%d_%H%M)_a$attempt.log
  if [ -f $RUN/weights/last.pt ]; then
    echo "=== $NAME attempt $attempt: resume (batch ${batch:-as saved}), epochs done=$(( $(wc -l < $RUN/results.csv) - 1 )) ($(date +%H:%M)) ==="
    python -u src/train.py --resume $RUN/weights/last.pt ${batch:+--batch $batch} > $LOG 2>&1
  else
    rm -rf $RUN  # a start that crashed before its first checkpoint
    echo "=== $NAME attempt $attempt: fresh start ($(date +%H:%M)) ==="
    python -u src/train.py --name $NAME "$@" > $LOG 2>&1
  fi
  rc=$?
  echo "=== $NAME attempt $attempt exit=$rc ($(date +%H:%M)) ==="
  [ $rc -eq 0 ] && { echo "TRAINING COMPLETE: $NAME"; break; }
  kill_leftovers
  if tr '\r' '\n' < $LOG | grep -qiE "out of memory|Only .* GB of GPU memory is free"; then
    fails=$((fails + 1)); [ $fails -ge 2 ] && batch=8
  else
    echo "non-memory failure, stopping for inspection"; break
  fi
  [ $attempt -ge 8 ] && { echo "too many attempts, stopping"; break; }
done
