#!/bin/bash
# Install SND Intelligence on a Mac without git.
# Warehouse is stored in ~/Library/Application Support/SND Intelligence
# so later updates do not require re-uploading history.
set -euo pipefail
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/fmcg-sales-intelligence-9302}"
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"
ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"

echo "Installing into ${APP_DIR}"
echo "Downloading ${ZIP_URL}"
mkdir -p /tmp/sndintel-dl
curl -L --fail "${ZIP_URL}" -o /tmp/sndintel-dl/app.zip
rm -rf /tmp/sndintel-dl/unpacked
mkdir -p /tmp/sndintel-dl/unpacked
unzip -q /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl/unpacked
SRC="$(find /tmp/sndintel-dl/unpacked -maxdepth 1 -type d -name 'SND-pro-*' | head -1)"
if [ -z "${SRC}" ]; then
  echo "Download did not contain the repo folder" >&2
  exit 1
fi
mkdir -p "${APP_DIR}"
# Replace code only. Never touch Application Support.
rsync -a --delete --exclude '.venv' "${SRC}/" "${APP_DIR}/"
cd "${APP_DIR}"
python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
echo
echo "Installed."
echo "Warehouse (do not delete): ~/Library/Application Support/SND Intelligence/warehouse.db"
echo
echo "Start the app:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
