#!/usr/bin/env sh
set -eu

export COMPASS_UI_HOST="${COMPASS_UI_HOST:-0.0.0.0}"
export COMPASS_UI_PORT="${COMPASS_UI_PORT:-5005}"

if [ "$#" -eq 0 ]; then
  exec python3 main.py --ui
fi

case "$1" in
  --*)
    exec python3 main.py --ui "$@"
    ;;
  *)
    exec "$@"
    ;;
esac
