#!/bin/bash
# Install SND Intelligence on a Mac without git. curl + .venv.
# Warehouse is stored in ~/Library/Application Support/SND Intelligence
# so later updates do not require re-uploading history.
set -euo pipefail
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/actionable-ops-layer-2f34}"
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"
ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"

echo "Installing into ${APP_DIR}"
rm -rf /tmp/sndintel-dl
mkdir -p /tmp/sndintel-dl "${APP_DIR}"
echo "Downloading ${ZIP_URL}"
curl -L --fail "${ZIP_URL}" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
if [ -z "${SRC}" ]; then
  echo "Download did not contain the repo folder" >&2
  exit 1
fi
# Replace code only. Never touch Application Support. Keep .venv if it exists.
rsync -a --delete --exclude '.venv' "${SRC}/" "${APP_DIR}/"
cd "${APP_DIR}"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e .
echo
echo "Installed."
echo "Warehouse (do not delete): ~/Library/Application Support/SND Intelligence/warehouse.db"
echo
echo "Start the app:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
