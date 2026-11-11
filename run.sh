#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -x bin/sysaudio-capture ]]; then
  ./build.sh
fi
if [[ -n "${PYTHON:-}" ]]; then
  PY="$PYTHON"
elif [[ -x .venv/bin/python ]]; then
  PY=".venv/bin/python"
elif [[ -x /usr/local/bin/python3.12 ]]; then
  PY="/usr/local/bin/python3.12"
else
  PY="$(command -v python3.12 || command -v python3)"
fi
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${1:-}" == "choose" ]]; then
  shift
  set -- --choose "$@"
elif [[ "${1:-}" == "web" ]]; then
  shift
  set -- --web "$@"
fi
exec "$PY" -m sysaudio_stt "$@"
