#!/bin/bash
# Update SND Intelligence on a Mac without git. Same curl + .venv method as
# the original install. Does not touch the warehouse.
#
#   bash ~/sndintel/scripts/mac_update.sh
set -euo pipefail
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/actionable-ops-layer-2f34}"
ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"

if [ ! -d "${APP_DIR}" ]; then
  echo "No app at ${APP_DIR}. Run scripts/mac_install.sh first." >&2
  exit 1
fi

rm -rf /tmp/sndintel-dl
mkdir -p /tmp/sndintel-dl
echo "Downloading ${ZIP_URL}"
curl -L --fail "${ZIP_URL}" -o /tmp/sndintel-dl/app.zip
unzip -o /tmp/sndintel-dl/app.zip -d /tmp/sndintel-dl
SRC="$(find /tmp/sndintel-dl -maxdepth 2 -type d -name 'SND-pro-*' | head -1)"
if [ -z "${SRC}" ]; then
  echo "ZIP did not contain the app folder." >&2
  exit 1
fi

echo "Updating code in ${APP_DIR}"
echo "Warehouse stays in ~/Library/Application Support/SND Intelligence"
rsync -a --delete --exclude '.venv' "${SRC}/" "${APP_DIR}/"
cd "${APP_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate
pip install -e .
echo
echo "Updated. Start with:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
echo "You do not need to re-upload closed months."
