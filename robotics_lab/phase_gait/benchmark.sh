#!/usr/bin/env bash
# Run the matching reference check and first phase-conditioned measurements.
set -euo pipefail
source /root/autodl-tmp/go2-ppo/go2-env.sh
cd "$GO2_ROOT"
base="$(cat forward-run.txt)/model_399.pt"
results="$GO2_ROOT/phase-results"
first="$results/trot-200"
kit=--/plugins/carb.tasking.plugin/threadCount=8

sample_resources() {
  {
    date
    cat /sys/fs/cgroup/memory.current
    head -1 /sys/fs/cgroup/{cpu,memory}.pressure
    for zone in /sys/class/thermal/thermal_zone*; do paste "$zone/type" "$zone/temp"; done
    nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used --format=csv
    df -h "$results"
  } >> "$results/resources.log"
}

evaluate() {
  local name="$1" checkpoint="$2"
  shift 2
  test -s "$checkpoint"
  sample_resources
  printf '%s\n' "$name" >> "$results/benchmark-progress.log"
  python -u phase_gait/evaluate.py --headless --kit_args="$kit" \
    --checkpoint "$checkpoint" --output "$results/$name" "$@" > "$results/$name.log" 2>&1
}

for attempt in {1..100}; do
  if grep -q "^Training seconds:" "$results/trot-200.log" &&
    ! pgrep -f '^python -u phase_gait/train.py .*trot-200$' >/dev/null; then
    break
  fi
  sleep 3
done
grep -q "^Training seconds:" "$results/trot-200.log"
test -s "$first/final.pt"
evaluate baseline-43 "$base" --baseline
evaluate trot-51-43 "$first/model_50.pt"
evaluate trot-200-43 "$first/final.pt"
evaluate trot-200-97 "$first/final.pt" --seed 97
printf '%s\n' done >> "$results/benchmark-progress.log"
