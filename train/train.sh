#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

tmux new -d -s train bash -c "
cd \"${SCRIPT_DIR}\" && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install -r \"${REPO_ROOT}/source/requirements.txt\" && \
export PYTHONPATH=\"${REPO_ROOT}:\${PYTHONPATH}\" && \
mkdir -p logs tmp/sb3_log && \
python train_scratch.py; \ # Either train_scratch.py or train_pretrained_critic.py
exec bash"

tmux attach -t train