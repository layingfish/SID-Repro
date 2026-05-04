set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${RUN_RQ1_ARGS:-}" && -z "${RUN_RQ2_ARGS:-}" && -z "${RUN_RQ3_ARGS:-}" && -z "${RUN_RQ4_ARGS:-}" ]]; then
  echo "Set RUN_RQ*_ARGS to run configured experiments, for example:"
  echo "  RUN_RQ1_ARGS='evaluate --pred ... --dataset ... --output ...' bash scripts/run_all.sh"
  echo
  echo "Available launchers:"
  echo "  bash scripts/run_data.sh --help"
  echo "  bash scripts/run_rq1.sh --help"
  echo "  bash scripts/run_rq2.sh --help"
  echo "  bash scripts/run_rq3.sh --help"
  echo "  bash scripts/run_rq4.sh --help"
  exit 0
fi

if [[ -n "${RUN_RQ1_ARGS:-}" ]]; then
  bash scripts/run_rq1.sh ${RUN_RQ1_ARGS}
fi
if [[ -n "${RUN_RQ2_ARGS:-}" ]]; then
  bash scripts/run_rq2.sh ${RUN_RQ2_ARGS}
fi
if [[ -n "${RUN_RQ3_ARGS:-}" ]]; then
  bash scripts/run_rq3.sh ${RUN_RQ3_ARGS}
fi
if [[ -n "${RUN_RQ4_ARGS:-}" ]]; then
  bash scripts/run_rq4.sh ${RUN_RQ4_ARGS}
fi
