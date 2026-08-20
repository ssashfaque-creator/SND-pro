#!/bin/bash
# Update SND Intelligence code on a Mac without git. Does not touch the warehouse.
set -euo pipefail
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/fmcg-sales-intelligence-9302}"
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"
ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"

if [ ! -d "${APP_DIR}" ]; then
  echo "No app at ${APP_DIR}. Run scripts/mac_install.sh first (or paste the install block from the chat)." >&2
  exit 1
fi

echo "Updating code in ${APP_DIR} from ${BRANCH}"
echo "Warehouse stays in ~/Library/Application Support/SND Intelligence"
mkdir -p /tmp/sndintel-dl
curl -L --fail "${ZIP_URL}" -o /tmp/sndintel-dl/app.zip
rm -rf /tmp/sndintel-dl/unpacked
mkdir -p /tmp/sndintel-dl/unpacked
unzip -q /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl/unpacked
SRC="$(find /tmp/sndintel-dl/unpacked -maxdepth 1 -type d -name 'SND-pro-*' | head -1)"
rsync -a --delete --exclude '.venv' --exclude '.streamlit/secrets.toml' "${SRC}/" "${APP_DIR}/"
cd "${APP_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e ".[dev]"
echo
echo "Updated. Start with:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
echo "You do not need to re-upload closed months."
