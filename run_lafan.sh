#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
mode=${1:-help}
if [[ "$mode" != train && "$mode" != evaluate && "$mode" != play ]]; then
  echo 'Usage: ./run_lafan.sh train|evaluate|play g1|bsrl [extra Python arguments]'
  echo 'Example: ./run_lafan.sh train bsrl --iterations 1000 --output outputs/bsrl_run'
  echo 'Example: ./run_lafan.sh evaluate g1 --checkpoint checkpoints/best/policy_best.pt --clips walk1_subject1'
  echo 'For headless video use evaluate with --video path.mp4; play requires a display.'
  exit 0
fi
shift
robot=g1
if [[ $# -gt 0 && "$1" != -* ]]; then
  robot=$1
  shift
fi
source ./env.sh
case "$robot" in
  g1) robot_id=g1_29dof; motion_dir=src/assets/motions/g1/lafan1_npz ;;
  bsrl) robot_id=bsrl_12dof; motion_dir=src/assets/motions/bsrl/lafan1_npz ;;
  *) echo "Unknown robot: $robot" >&2; exit 2 ;;
esac
if [[ "$mode" == train ]]; then
  exec python scripts/train_supermimic.py --robot "$robot_id" --device cuda:0 \
    --num-envs 128 --capacity 256 --wm-batch-size 128 --policy-batch-size 128 \
    --no-compile --amp none "$@"
else
  exec python "scripts/${mode}_supermimic.py" --robot "$robot_id" --device cuda:0 --motion-dir "$motion_dir" "$@"
fi
