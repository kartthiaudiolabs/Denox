#!/usr/bin/env bash
# DENOX Mac App Launcher

DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$DIR" || exit 1

echo "=========================================================="
echo "      DENOX // QUANTUM AUDIO STUDIO (macOS)              "
echo "  Apple Metal GPU (MPS) Acceleration · 48 kHz Master Clock"
echo "=========================================================="

if [ ! -d ".venv" ]; then
    echo "Virtual environment .venv not found. Setting up..."
    uv venv --python 3.12 .venv
    uv pip install -e ".[demo]"
    uv pip install pywebview
fi

exec .venv/bin/python -m mac_app.main "$@"
