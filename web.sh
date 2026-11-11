#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
exec ./run.sh --web "$@"
