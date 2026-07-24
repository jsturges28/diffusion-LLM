#!/usr/bin/env bash
#
# Install a desktop launcher (app-menu entry) for the Diffusion LLM
# Visualizer on Linux. The .desktop file carries machine-specific
# absolute paths (venv Python, repo directory, icon), so it is
# generated here rather than committed to the repo.
#
# Usage: scripts/install_desktop_entry.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
LAUNCHER="$REPO_ROOT/desktop.py"
ICON_PATH="$REPO_ROOT/assets/icon.svg"
APPS_DIR="$HOME/.local/share/applications"
ENTRY_PATH="$APPS_DIR/llm-xai-visualizer.desktop"

if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: venv Python not found at $VENV_PYTHON" >&2
  echo "Create the environment first (see README Setup)." >&2
  exit 1
fi

if [ ! -f "$LAUNCHER" ]; then
  echo "Error: desktop.py not found at $LAUNCHER" >&2
  exit 1
fi

mkdir -p "$APPS_DIR"

cat > "$ENTRY_PATH" <<EOF
[Desktop Entry]
Type=Application
Name=LLM XAI Visualizer
Comment=Local visual playground for discrete diffusion LLMs
Exec=$VENV_PYTHON $LAUNCHER
Path=$REPO_ROOT
Icon=$ICON_PATH
Terminal=false
Categories=Development;Science;Utility;
StartupWMClass=llm-xai-visualizer
EOF

chmod 644 "$ENTRY_PATH"
echo "Installed desktop entry: $ENTRY_PATH"

if command -v update-desktop-database >/dev/null 2>&1; then
  update-desktop-database "$APPS_DIR" >/dev/null 2>&1 || true
fi

echo "Done. Look for \"LLM XAI Visualizer\" in your app menu."
