#!/bin/bash
# Update SND Intelligence on a Mac without git. Does not touch the warehouse.
#
# Preferred: download the ZIP in the browser (GitHub → Code → Download ZIP)
# then pass it, or drop it in Downloads as SND-pro*.zip
#
#   bash ~/sndintel/scripts/mac_update.sh ~/Downloads/SND-pro-cursor-actionable-ops-layer-2f34.zip
set -euo pipefail
APP_DIR="${SNDINTEL_APP_DIR:-$HOME/sndintel}"
REPO="${SNDINTEL_GITHUB_REPO:-ssashfaque-creator/SND-pro}"
BRANCH="${SNDINTEL_APP_BRANCH:-cursor/actionable-ops-layer-2f34}"

if [ ! -d "${APP_DIR}" ]; then
  echo "No app at ${APP_DIR}. Run scripts/mac_install.sh first." >&2
  exit 1
fi

ZIP="${1:-}"
if [ -z "${ZIP}" ]; then
  ZIP="$(ls -t "${HOME}/Downloads"/SND-pro*.zip 2>/dev/null | head -1 || true)"
fi

mkdir -p /tmp/sndintel-dl
if [ -n "${ZIP}" ] && [ -f "${ZIP}" ]; then
  echo "Using ZIP ${ZIP}"
  cp "${ZIP}" /tmp/sndintel-dl/app.zip
else
  ZIP_URL="https://github.com/${REPO}/archive/refs/heads/${BRANCH}.zip"
  echo "No Downloads ZIP found. Trying ${ZIP_URL}"
  echo "Private repo: download ZIP in the browser while logged in, then rerun with that file."
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
  echo "ZIP did not contain the app folder." >&2
  exit 1
fi

echo "Updating code in ${APP_DIR}"
echo "Warehouse stays in ~/Library/Application Support/SND Intelligence"
rsync -a --delete --exclude '.venv' --exclude 'data' --exclude '.git' --exclude '.streamlit/secrets.toml' "${SRC}/" "${APP_DIR}/"
cd "${APP_DIR}"
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install -e .
echo
echo "Updated. Start with:"
echo "  cd ${APP_DIR} && source .venv/bin/activate && snd-intel app"
echo "You do not need to re-upload closed months."
