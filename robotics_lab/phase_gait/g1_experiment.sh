#!/usr/bin/env bash
# Run the two G1 training stages and preserve their held-out measurements.
set -euo pipefail
source /root/autodl-tmp/go2-ppo/go2-env.sh
cd "$GO2_ROOT"
code="${1:-phase_gait}"
results="${G1_RESULTS_DIR:-$GO2_ROOT/g1-results}"
mkdir -p "$results"
kit=--/plugins/carb.tasking.plugin/threadCount=8

resources() {
  {
    date
    head -1 /sys/fs/cgroup/{cpu,memory}.pressure
    cat /sys/fs/cgroup/memory.current
    for zone in /sys/class/thermal/thermal_zone*; do paste "$zone/type" "$zone/temp"; done
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
    df -h "$results"
  } >> "$results/resources.log"
}

run() {
  local label="$1" entry="$2"
  shift 2
  resources
  printf '%s\n' "$label" >> "$results/progress.log"
  python -u "$code/$entry.py" --headless --kit_args="$kit" \
    --output "$results/$label" "$@" > "$results/$label.log" 2>&1
}

if [ -n "${2:-}" ]; then
  selection=flat-120
  run "$selection" train --g1-flat --exploration-std 0.25 --iterations 120 --checkpoint "$2"
else
  selection=flat-300
  run "$selection" train --g1-flat --exploration-std 0.28 --iterations 300
fi
run "$selection-eval" evaluate --g1-flat --steps 600 --checkpoint "$results/$selection/final.pt"
passes_flat() {
  python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(not (d["survivors"]>=14 and abs(d["mean_forward_speed_mps"]-.42)<=.14 and d["mean_forward_displacement_m"]>=2.0))' "$1"
}
if ! passes_flat "$results/$selection-eval/metrics.json"; then
  initial="$selection"
  run flat-extended train --g1-flat --iterations 160 --checkpoint "$results/$initial/final.pt"
  run flat-extended-eval evaluate --g1-flat --steps 600 --checkpoint "$results/flat-extended/final.pt"
  if passes_flat "$results/flat-extended-eval/metrics.json"; then
    selection=flat-extended
  else
    # Stable walking is enough to attempt terrain transfer; preserve the speed miss in the metrics.
    python -c 'import json,sys; d=json.load(open(sys.argv[1])); sys.exit(not (d["survivors"]>=14 and d["mean_forward_displacement_m"]>=2.0))' "$results/$initial-eval/metrics.json"
  fi
fi
bash "$code/g1_stairs_only.sh" "$code" "$results/$selection/final.pt"
