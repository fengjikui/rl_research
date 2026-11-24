#!/usr/bin/env bash
# Continue from a measured stable flat G1 policy.
set -euo pipefail
source /root/autodl-tmp/go2-ppo/go2-env.sh
cd "$GO2_ROOT"
code="${1:-phase_gait}"
base="$2"
results="${G1_RESULTS_DIR:-$GO2_ROOT/g1-results}"
mkdir -p "$results"
kit=--/plugins/carb.tasking.plugin/threadCount=8

run() {
  local name="$1" entry="$2"
  shift 2
  {
    date
    cat /sys/fs/cgroup/memory.current
    head -1 /sys/fs/cgroup/{cpu,memory}.pressure
    for zone in /sys/class/thermal/thermal_zone*; do paste "$zone/type" "$zone/temp"; done
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
  } >> "$results/resources.log"
  printf '%s\n' "$name" >> "$results/progress.log"
  python -u "$code/$entry.py" --headless --kit_args="$kit" \
    --output "$results/$name" "$@" > "$results/$name.log" 2>&1
}

run flat-selected-video evaluate --g1-flat --steps 550 --num_envs 1 --video --checkpoint "$base"
run stairs-240 train --g1-stairs --checkpoint "$base" --iterations 240 \
  --exploration-std 0.18 --fixed-lr 0.0003
run stairs-240-eval evaluate --g1-stairs --steps 600 --checkpoint "$results/stairs-240/final.pt"
selection=stairs-240
if ! python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(not d["climbed_all_risers"]>=14)' "$results/stairs-240-eval/metrics.json"; then
  run stairs-390 train --g1-stairs --iterations 150 --checkpoint "$results/stairs-240/final.pt" --fixed-lr 0.0003
  selection=stairs-390
  run stairs-390-eval evaluate --g1-stairs --steps 600 --checkpoint "$results/stairs-390/final.pt"
fi
printf '%s\n' "$selection" > "$results/stairs-selected.txt"
run stairs-video evaluate --g1-stairs --steps 550 --num_envs 1 --video --checkpoint "$results/$selection/final.pt"
printf '%s\n' done >> "$results/progress.log"
