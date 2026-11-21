#!/usr/bin/env bash
# 6 槽 x 每槽 256K 满血: 1/2/4/6 路实际并发扫描
set -u
cd /c/Work/LLM/Basalt || exit 1
KEY=$(python keys-arg.py . | cut -d, -f1)
LOG=bench_6x256k_sweep.log
: > "$LOG"
for n in 1 2 4 6; do
  echo "##### concurrency=$n  $(date +%H:%M:%S)" | tee -a "$LOG"
  .venv/Scripts/python bench_slots.py --key "$KEY" --concurrency "$n" \
      --prompt-chars 787000 --max-tokens 16384 >> "$LOG" 2>&1
  echo "exit=$?" | tee -a "$LOG"
done
echo "ALL DONE $(date +%H:%M:%S)" >> "$LOG"
