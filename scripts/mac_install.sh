#!/bin/bash
# Install SND Intelligence on a Mac without git.
# Warehouse is stored in ~/Library/Application Support/SND Intelligence
# so later updates do not require re-uploading history.
#
# If GitHub is private, download the ZIP in the browser first:
#   bash scripts/mac_install.sh ~/Downloads/SND-pro-cursor-actionable-ops-layer-2f34.zip
set -euo pipefail
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/actionable-ops-layer-2f34}"
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"

ZIP="${1:-}"
if [ -z "${ZIP}" ]; then
  ZIP="$(ls -t "${HOME}/Downloads"/SND-pro*.zip 2>/dev/null | head -1 || true)"
fi

echo "Installing into ${APP_DIR}"
mkdir -p /tmp/sndintel-dl
if [ -n "${ZIP}" ] && [ -f "${ZIP}" ]; then
  echo "Using ZIP ${ZIP}"
  cp "${ZIP}" /tmp/sndintel-dl/app.zip
else
  ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"
  echo "Downloading ${ZIP_URL}"
  AUTH=()
  if [ -n "${GH_TOKEN:-}" ]; then
    AUTH=(-H "Authorization: Bearer ${GH_TOKEN}" -H "Accept: application/vnd.github+json")
  fi
  curl -L --fail "${AUTH[@]}" "${ZIP_URL}" -o /tmp/sndintel-dl/app.zip
fi
rm -rf /tmp/sndintel-dl/unpacked
mkdir -p /tmp/sndintel-dl/unpacked
unzip -q /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl/unpacked
SRC="$(find /tmp/sndintel-dl/unpacked -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
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
python -m pip install -e .
echo
echo "Installed."
echo "Warehouse (do not delete): ~/Library/Application Support/SND Intelligence/warehouse.db"
echo
echo "Start the app:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
