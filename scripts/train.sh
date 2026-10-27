#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
out="${OUT:-outputs/ppo-v2}"
if [[ -e "$out" ]]; then
  echo "Output already exists: $out. Use a new OUT for a fresh experiment." >&2
  exit 1
fi
python -m kart.ppo \
  --device "${DEVICE:-cuda}" --threads 4 --gpu-memory-gb "${GPU_MEMORY_GB:-2}" \
  --params configs/train-params.json --pure-ppo --balanced-scenes \
  --seed 20261031 --envs "${ENVS:-256}" --horizon 64 --segments 16 --max-time 60 \
  --iters "${ITERS:-2000}" --lr .0003 --loop-delay 0 0 --save-every 10 \
  --case configs/level645-case.json --case-probability .15 \
  --target-kl .015 --frame-skip-max 3 --timing --deterministic \
  --ent-final .001 --ent-anneal-iters 1000 \
  --validation-envs 128 --eval-every 50 --early-stop-patience 16 \
  --early-stop-min-iters 1000 --early-stop-delta .0001 --out "$out"
